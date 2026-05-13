# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import util.misc as misc
from util.datasets import build_fmow_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.timm import add_weight_decay

# import models_mae
import models_mae
import models_mae_group_channels
import models_mae_temporal

from engine_pretrain import train_one_epoch, train_one_epoch_temporal

from functools import partial
from vit_lora_util import Block as LoraBlock
from train_lora_util import activate_lora, is_merged, lora_merge_all, deactivate_lora, delete_qkv


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type", default=None, choices=["group_c", "temporal", "vanilla"], help="Use channel model"
    )
    parser.add_argument(
        "--model", default="mae_vit_large_patch16", type=str, metavar="MODEL", help="Name of model to train"
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--patch_size", default=16, type=int, help="images input size")

    parser.add_argument(
        "--mask_ratio", default=0.75, type=float, help="Masking ratio (percentage of removed patches)."
    )
    parser.add_argument(
        "--spatial_mask",
        action="store_true",
        default=False,
        help="Whether to mask all channels of a spatial location. Only for indp c model",
    )

    parser.add_argument(
        "--norm_pix_loss", action="store_true", help="Use (per-patch) normalized pixels as targets for computing loss"
    )
    parser.set_defaults(norm_pix_loss=False)

    # LoRA info
    parser.add_argument(
        "--lora_type",
        default="lora",
        choices=["lora", "monarch", "monarch_mult", "boft", "butterfly", "butterfly_mult"],
        help="Which matrix decomposition type to use.",
    )
    parser.add_argument("--load_weights", default="", help="pretrain from checkpoint (possibly other domain)")
    parser.add_argument("--lora_rank", type=int, default=8, help="Lora matrices rank")
    parser.add_argument("--decoder_lora_rank", type=int, default=None, help="MAE Decoder Lora matrices rank")
    parser.add_argument("--lora_layers", type=str, default=["attn"], nargs="+", help="Layers to use lora on")
    parser.add_argument(
        "--unfreeze_blocks", type=int, default=None, nargs="+", help="Which ViT blocks to unfreeze to train fully."
    )
    parser.add_argument(
        "--unfreeze_embed", action="store_true", default=False, help="Whether to train patch embed. Default is false."
    )
    parser.add_argument(
        "--unfreeze_cls_token",
        action="store_true",
        default=False,
        help="Whether to train cls_token. Default is false.",
    )
    parser.add_argument(
        "--unfreeze_norm", action="store_true", default=False, help="Whether to train norms. Default is false."
    )
    parser.add_argument(
        "--freeze_decoder", action="store_true", default=False, help="Don't LoRA train the decoder attn weights."
    )
    parser.add_argument(
        "--lora_attn_key", action="store_true", default=False, help="Also use lora on key proj weights."
    )
    parser.add_argument(
        "--lora_attn_proj",
        action="store_true",
        default=False,
        help="Also use lora on attn proj weights (used after qkv).",
    )
    parser.add_argument("--unfreeze_all", default=False, action="store_true")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.00, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=0, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument(
        "--train_path", default="data_and_checkpoints/fmow_csvs/train_62classes.csv", type=str, help="Train .csv path"
    )
    parser.add_argument(
        "--dataset_type",
        default="rgb",
        choices=[
            "rgb",
            "temporal",
            "sentinel",
            "euro_sat",
            "naip",
            "camelyon17",
            "globalwheat",
            "iwildcam",
            "poverty",
            "rxrx1",
            "globalwheat_unlabeled",
            "camelyon17_unlabeled",
        ],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--masked_bands",
        type=int,
        nargs="+",
        default=None,
        help="Sequence of band indices to mask (with mean val) in sentinel dataset",
    )
    parser.add_argument(
        "--dropped_bands",
        type=int,
        nargs="+",
        default=None,
        help="Which bands (0 indexed) to drop from sentinel data.",
    )
    parser.add_argument(
        "--grouped_bands", type=int, nargs="+", action="append", default=[], help="Bands to group for GroupC mae"
    )

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=5, help="How frequently (in epochs) to save ckpt")
    parser.add_argument("--wandb", type=str, default=None, help="Wandb project name, eg: sentinel_pretrain")
    parser.add_argument(
        "--wandb_logdir",
        type=str,
        default="./wandb",
        help="Where the wandb log info is stored. Warning: takes a lot of space fast",
    )
    parser.add_argument("--wandb_entity", type=str, default="FILL IN", help="Wandb entity name")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int)  # prev default was -1
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


@record
def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_fmow_dataset(is_train=True, args=args)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    ## Set the type of matrix factorization
    block_type = LoraBlock
    lora_attr_str = "lora_A"

    is_merged_f = partial(is_merged, key=lora_attr_str)
    merge_all = partial(lora_merge_all, unmerge=False, key=lora_attr_str)
    unmerge_all = partial(lora_merge_all, unmerge=True, key=lora_attr_str)

    # define the model
    if args.model_type == "group_c":
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
        print(f"Grouping bands {args.grouped_bands}")
        model = models_mae_group_channels.__dict__[args.model](
            block_type=block_type,
            img_size=args.input_size,
            patch_size=args.patch_size,
            in_chans=dataset_train.in_c,
            channel_groups=args.grouped_bands,
            spatial_mask=args.spatial_mask,
            norm_pix_loss=args.norm_pix_loss,
        )
    elif args.model_type == "temporal":
        model = models_mae_temporal.__dict__[args.model](
            block_type=block_type,
            img_size=args.input_size,
            patch_size=args.patch_size,
            in_chans=dataset_train.in_c,
            norm_pix_loss=args.norm_pix_loss,
        )
    # non-spatial, non-temporal
    else:
        model = models_mae.__dict__[args.model](
            block_type=block_type,
            img_size=args.input_size,
            patch_size=args.patch_size,
            in_chans=dataset_train.in_c,
            norm_pix_loss=args.norm_pix_loss,
        )

    ## Set up lora

    if args.load_weights:
        checkpoint = torch.load(args.load_weights, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.load_weights)
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()

        # if 'patch_embed.proj.weight' in checkpoint_model and 'patch_embed.proj.weight' in state_dict:
        #     ckpt_patch_embed_weight = checkpoint_model['patch_embed.proj.weight']
        #     model_patch_embed_weight = state_dict['patch_embed.proj.weight']
        #     if ckpt_patch_embed_weight.shape[1] != model_patch_embed_weight.shape[1]:
        #         print('Using 3 channels of ckpt patch_embed')
        #         model.patch_embed.proj.weight.data[:, :3, :, :] = ckpt_patch_embed_weight.data[:, :3, :, :]

        # TODO: Do something smarter? Should pos_embed be here?
        for k in [
            "pos_embed",
            "decoder_pos_embed",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "patch_embed.0.proj.weight",
            "patch_embed.0.proj.bias",
            "patch_embed.1.proj.weight",
            "patch_embed.1.proj.bias",
            "patch_embed.2.proj.weight",
            "patch_embed.2.proj.bias",
            "decoder_pred.0.weight",
            "decoder_pred.0.bias",
            "decoder_pred.1.weight",
            "decoder_pred.1.bias",
            "decoder_pred.2.weight",
            "decoder_pred.2.bias",
        ]:
            if k not in state_dict and k not in checkpoint_model:
                continue

            if k not in state_dict or (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # TODO: change assert msg based on patch_embed
        print(set(msg.missing_keys))

    if args.decoder_lora_rank is None:
        args.decoder_lora_rank = args.lora_rank

    if not args.unfreeze_all:
        if args.unfreeze_blocks is not None:
            lora_blocks = [idx for idx in list(range(len(model.blocks))) if idx not in args.unfreeze_blocks]
            for block_idx in lora_blocks:
                activate_lora(
                    model.blocks[block_idx],
                    args.lora_layers,
                    args.lora_rank,
                    include_attn_key=args.lora_attn_key,
                    include_attn_proj=args.lora_attn_proj,
                )
            if not args.freeze_decoder:
                activate_lora(
                    model.decoder_blocks,
                    args.lora_layers,
                    args.decoder_lora_rank,
                    include_attn_key=args.lora_attn_key,
                    include_attn_proj=args.lora_attn_proj,
                )
        else:
            activate_lora(
                model.blocks,
                args.lora_layers,
                args.lora_rank,
                include_attn_key=args.lora_attn_key,
                include_attn_proj=args.lora_attn_proj,
            )
            activate_lora(
                model.decoder_blocks,
                args.lora_layers,
                args.decoder_lora_rank,
                include_attn_key=args.lora_attn_key,
                include_attn_proj=args.lora_attn_proj,
            )

        for name, param in model.named_parameters():
            if not (
                args.lora_type in name
                or (args.unfreeze_embed and ("patch_embed" in name or "decoder_pred" in name))
                or (args.unfreeze_norm and "norm" in name)
                or (args.unfreeze_cls_token and "cls_token" in name)
                or (
                    args.unfreeze_blocks is not None
                    and any([f"blocks.{idx}." in name and "decoder" not in name for idx in args.unfreeze_blocks])
                )
            ):
                param.requires_grad_(False)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.4f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Need to unmerge lora before resuming pretraining
    if args.resume:
        unmerge_all(model)

    # Set up wandb
    if misc.is_main_process() and args.wandb is not None:
        wandb.init(project=args.wandb, dir=args.wandb_logdir, entity=args.wandb_entity)
        wandb.config.update(args)
        wandb.watch(model)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.model_type == "temporal":
            train_stats = train_one_epoch_temporal(
                model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
            )
        else:
            train_stats = train_one_epoch(
                model,
                model_without_ddp,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer=log_writer,
                args=args,
            )

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            if args.lora_rank > 0:
                model.train(False)  # To merge lora weights
                assert is_merged_f(model)
                if args.unfreeze_blocks is not None:
                    lora_blocks = [
                        idx for idx in list(range(len(model_without_ddp.blocks))) if idx not in args.unfreeze_blocks
                    ]
                    for block_idx in lora_blocks:
                        deactivate_lora(
                            model_without_ddp.blocks[block_idx],
                            activate_layers=args.lora_layers,
                            delete_separate_proj=False,
                        )
                    if not args.freeze_decoder:
                        deactivate_lora(
                            model_without_ddp.decoder_blocks,
                            activate_layers=args.lora_layers,
                            delete_separate_proj=False,
                        )
                else:
                    deactivate_lora(
                        model, activate_layers=args.lora_layers, delete_separate_proj=False
                    )  # So that qkv is created
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
            if args.lora_rank > 0:
                # Don't need qkv so delete
                if args.unfreeze_blocks is not None:
                    lora_blocks = [
                        idx for idx in list(range(len(model_without_ddp.blocks))) if idx not in args.unfreeze_blocks
                    ]
                    for block_idx in lora_blocks:
                        delete_qkv(model_without_ddp.blocks[block_idx], layers=args.lora_layers)
                    if not args.freeze_decoder:
                        delete_qkv(model_without_ddp.decoder_blocks, layers=args.lora_layers)
                else:
                    delete_qkv(model, layers=args.lora_layers)
                model.train(True)
                assert not is_merged_f(model)

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            try:
                wandb.log(log_stats)
            except ValueError:
                print(f"Invalid stats?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
