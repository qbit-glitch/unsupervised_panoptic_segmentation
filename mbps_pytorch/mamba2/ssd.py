# Copyright (c) 2024, Albert Gu and Tri Dao.
# Pure PyTorch SSD (State Space Duality) implementation.
# Ported from official state-spaces/mamba v2.3.0 reference functions.
# No Triton, no CUDA — works on CPU, MPS, and CUDA.

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Segment sum (from ssd_minimal.py)
# ---------------------------------------------------------------------------

def segsum(x):
    """More stable segment sum calculation.

    Computes the cumulative segment sums needed for the SSD recurrence.
    This is the numerically stable version from the Mamba2 paper (Listing 1).
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


# ---------------------------------------------------------------------------
# Minimal SSD (Listing 1 from the paper — ssd_minimal.py)
# ---------------------------------------------------------------------------

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """Minimal SSD implementation (Listing 1 from the Mamba2 paper).

    This is the simplest form of SSD — takes pre-discretized inputs.
    For the full pipeline (with dt, softplus, D, z), use ssd_chunk_scan_combined.

    Arguments:
        X: (batch, length, n_heads, d_head) — input * dt
        A: (batch, length, n_heads) — A * dt (pre-discretized)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: chunk size for the block-decomposition
        initial_states: optional (batch, 1, n_heads, d_head, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


# ---------------------------------------------------------------------------
# Component functions (from ssd_chunk_state.py, ssd_state_passing.py, ssd_chunk_scan.py)
# ---------------------------------------------------------------------------

def chunk_state(B, x, dt, dA_cumsum):
    """Compute SSM state for each chunk.

    From ssd_chunk_state.py:chunk_state_ref.

    Arguments:
        B: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)


def state_passing(states, dA_chunk_cumsum, initial_states=None):
    """Pass states across chunks via weighted cumulative sum.

    From ssd_state_passing.py:state_passing_ref.

    Arguments:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    # (batch, nheads, nchunks, nchunks)
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    # (batch, nheads, nchunks, nchunks) — clamp to prevent gradient explosion
    # from upper triangle (positive differences → huge exp, masked later but
    # gradients still flow through unmasked exp)
    decay_chunk = torch.exp(dt_chunk_segment_sum.clamp(max=0))
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]


def chunk_scan(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """Compute output for each chunk given previous states.

    From ssd_chunk_scan.py:chunk_scan_ref.

    Arguments:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    # Clamp upper triangle to prevent gradient explosion (same fix as GDN)
    decay = torch.exp(dt_segment_sum.clamp(max=0))
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)


# ---------------------------------------------------------------------------
# Combined SSD (from ssd_combined.py:ssd_chunk_scan_combined_ref)
# ---------------------------------------------------------------------------

def ssd_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None,
                            dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")),
                            return_final_states=False):
    """Full SSD scan with dt processing, chunking, and optional D/z.

    This is the main entry point for training — equivalent to
    mamba_chunk_scan_combined from the official Triton implementation.

    Arguments:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads,)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,) or None
        z: (batch, seqlen, nheads, headdim) or None
        dt_bias: (nheads,) or None
        dt_softplus: bool
        dt_limit: (float, float) — clamp dt values
        return_final_states: bool
    Return:
        out: (batch, seqlen, nheads, headdim)
        final_states: (batch, nheads, headdim, dstate) — only if return_final_states
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]

    # Pad sequence to multiple of chunk_size
    pad_len = 0
    if seqlen % chunk_size != 0:
        pad_len = chunk_size - seqlen % chunk_size
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        if z is not None:
            z = F.pad(z, (0, 0, 0, 0, 0, pad_len))

    padded_seqlen = x.shape[1]

    # Process dt: reshape, add bias, softplus, clamp
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # High precision for cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    if dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    # Compute dA_cumsum
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)

    # 1. Compute the state for each chunk
    states = chunk_state(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)

    # 2. Pass the state to all the chunks by weighted cumsum
    states = rearrange(state_passing(
        rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1]
    )[0], "... (p n) -> ... p n", n=dstate)
    states = states.to(states_dtype)

    # 3. Compute the output for each chunk
    out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)

    # Remove padding
    if pad_len > 0:
        out = out[:, :seqlen]

    if return_final_states:
        # Compute final states by running state_passing and taking the last
        # We need to recompute to get the final state
        states_for_final = chunk_state(B, x, dt, dA_cumsum)
        if states_for_final.dtype not in [torch.float32, torch.float64]:
            states_for_final = states_for_final.to(torch.float32)
        _, final_states = state_passing(
            rearrange(states_for_final, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1]
        )
        final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate)
        return out, final_states

    return out
