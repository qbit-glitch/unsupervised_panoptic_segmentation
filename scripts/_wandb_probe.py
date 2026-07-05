"""Find broken dist metadata + test wandb.init in the current WANDB_MODE (under trainer PYTHONPATH)."""
import os
import importlib.metadata as M

bad = []
for d in M.distributions():
    try:
        meta = d.metadata
        nm = meta["Name"] if meta is not None else None
        if nm is None:
            bad.append(str(getattr(d, "_path", d)))
    except Exception as e:
        bad.append(f"{getattr(d, '_path', d)} ERR {type(e).__name__}: {e}")
print("N_BAD_META", len(bad))
for b in bad[:15]:
    print("  BAD:", b)

mode = os.environ.get("WANDB_MODE", "?")
try:
    import wandb
    r = wandb.init(project="probe", dir="/tmp")
    print(f"WANDB_MODE={mode} INIT_OK {type(r).__name__}")
    r.finish()
except Exception as e:
    print(f"WANDB_MODE={mode} INIT_FAIL {type(e).__name__}: {e}")
