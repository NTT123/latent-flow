# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is based on
# https://github.com/karpathy/nanoGPT/blob/master/train.py

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import inspect
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import ot
import torch
from diffusers import AutoencoderDC
from torch.nn.attention.flex_attention import create_block_mask
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench

import wandb
from cfm import create_cfm

# from mix_mlp_model import DiT
from model import DiT
from sample import sample


def configure_optimizers(
    model, weight_decay, learning_rate, betas, device_type, logger
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": learning_rate},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"Use fused AdamW: {use_fused}")

    return optimizer


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if RANK == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# CUDA setup
assert torch.cuda.is_available()

# DDP setup
RANK = 0
DEVICE = 0
WORLD_SIZE = 1
IS_DISTRIBUTED = False
if "LOCAL_RANK" in os.environ:  # torchrun setup
    dist.init_process_group("nccl")
    RANK = dist.get_rank()
    DEVICE = RANK % torch.cuda.device_count()
    WORLD_SIZE = dist.get_world_size()
    IS_DISTRIBUTED = True


# Parse config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
with open(args.config, "r") as f:
    CONFIG = yaml.safe_load(f)

model_config = CONFIG["model"]
opt_config = CONFIG["optimization"]
data_config = CONFIG["data"]
training_config = CONFIG["training"]
wandb_config = training_config["wandb"]

# Validate batch size and set seed
assert opt_config["global_batch_size"] % WORLD_SIZE == 0
seed = training_config["seed"] * WORLD_SIZE + RANK
torch.manual_seed(seed)
torch.cuda.set_device(DEVICE)

# Initialize wandb if enabled
if RANK == 0 and wandb_config["enable"]:
    wandb.init(project=wandb_config["project"], config=CONFIG)

# Setup experiment directories and logger
if RANK == 0:
    os.makedirs(training_config["results_dir"], exist_ok=True)
    experiment_index = len(glob(f"{training_config['results_dir']}/*"))
    model_string_name = "anime-latent-flow"
    experiment_dir = (
        f"{training_config['results_dir']}/{experiment_index:03d}-{model_string_name}"
    )
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
else:
    logger = create_logger(None)

# Log config and setup info
logger.info("Config:")
for key, value in CONFIG.items():
    logger.info(f"  {key}: {value}")
logger.info(f"Starting rank={RANK}, seed={seed}, world_size={WORLD_SIZE}.")


def get_batch(step, batch_size):
    # Load dataset from memmap file
    data_dim = model_config["in_channels"]
    input_size = model_config["input_size"]
    data = torch.load(
        data_config["data_path"], map_location="cpu", weights_only=True, mmap=True
    )
    # Create random number generator
    seed = step * WORLD_SIZE + RANK
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    indices = rng.choice(data.shape[0], size=batch_size, replace=False).astype(np.int64)
    # Create batch data array
    batch_data = torch.empty(
        batch_size, data_dim, input_size, input_size, dtype=torch.float32, device="cpu"
    )
    # Fill batch data one sequence at a time
    for i, idx in enumerate(indices):
        batch_data[i] = data[idx]
    x = batch_data.to(device=DEVICE, non_blocking=True)
    if data_config["normalize"]:
        x = (x - data_config["data_mean"]) / data_config["data_std"]
    x = x.permute(0, 2, 3, 1)  # N H W C
    return x


# Initialize model
model = DiT(**model_config).float()

# log model architecture
logger.info(f"Model architecture: {model}")

# Setup model and EMA
ema = deepcopy(model).to(DEVICE)
requires_grad(ema, False)
model = model.to(DEVICE)
simple_model = model


if IS_DISTRIBUTED:
    model = DDP(model, device_ids=[RANK])

use_compile = training_config.get("enable_compile", False)
if use_compile:
    model = torch.compile(model, dynamic=False)
logger.info(f"Use torch.compile: {use_compile}")


update_ema(ema, simple_model, decay=0)
cfm = create_cfm()
logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Configure optimizer
opt = configure_optimizers(
    model=model,
    weight_decay=opt_config["weight_decay"],
    learning_rate=opt_config["learning_rate"],
    betas=(opt_config["betas"]["beta1"], opt_config["betas"]["beta2"]),
    device_type="cuda",
    logger=logger,
)

# Load checkpoint if resuming training
train_steps = 0
ckpt_path = training_config.get("resume_from_ckpt", None)
if ckpt_path:
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(
            ckpt_path, map_location=lambda storage, loc: storage, weights_only=False
        )
        checkpoint["model"]["pos_embed"] = torch.clone(simple_model.pos_embed)
        checkpoint["ema"]["pos_embed"] = torch.clone(simple_model.pos_embed)
        simple_model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        train_steps = checkpoint["train_steps"]
        logger.info(f"Resuming from step {train_steps}")
        torch.manual_seed(seed + train_steps)
    else:
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")


# Prepare models for training
model.train()
ema.eval()

# Initialize training monitoring variables
log_steps = 0
running_loss = 0
start_time = time()


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < opt_config["warmup_iters"]:
        return opt_config["learning_rate"] * (it + 1) / (opt_config["warmup_iters"] + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > opt_config["lr_decay_iters"]:
        return opt_config["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - opt_config["warmup_iters"]) / (
        opt_config["lr_decay_iters"] - opt_config["warmup_iters"]
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return opt_config["min_lr"] + coeff * (
        opt_config["learning_rate"] - opt_config["min_lr"]
    )


def compute_loss(model, x0, x1):
    """
    Compute loss for a batch of data
    """
    t = torch.rand(x0.shape[0], device=x0.device, dtype=torch.float32)
    loss_dict = cfm.training_losses(flow=model, x_0=x0, x_1=x1, t=t, model_kwargs={})
    loss = loss_dict["loss"].float().mean()
    return loss


use_bfloat16 = training_config.get("use_bfloat16", False)
logger.info(f"Use bfloat16: {use_bfloat16}")

logger.info(f"Use learning rate decay: {opt_config['decay_lr']}")
batch_size = opt_config["global_batch_size"] // WORLD_SIZE


def load_decoder():
    ae_model_name = CONFIG["ae_model_name"]
    ae = (
        AutoencoderDC.from_pretrained(ae_model_name, torch_dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    @torch.compile
    def decoder_(x):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
                x = x * data_config["data_std"] + data_config["data_mean"]
                y = ae.decode(x).sample
                y = y * 0.5 + 0.5
                return y

    return decoder_


logger.info(f"Loading decoder from {CONFIG['ae_model_name']}")
decoder = load_decoder()


def generate_ot_pairs(x1, seed=0):
    n = x1.shape[0]
    # create numpy random generator
    # rng = np.random.Generator(np.random.PCG64(seed=seed))
    # x0 = rng.normal(size=x1.shape).astype(np.float32)
    x0 = torch.randn_like(x1)
    d1 = x1.reshape(n, -1)
    d0 = x0.reshape(n, -1)
    # loss matrix
    M = ot.dist(d0, d1)
    a, b = torch.ones((n,), dtype=torch.float32, device=DEVICE), torch.ones(
        (n,), dtype=torch.float32, device=DEVICE
    )
    G0 = ot.emd(a, b, M, numThreads=8)
    d1 = torch.matmul(G0, d1)
    x1 = d1.reshape(*x1.shape)
    return x0, x1


while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(train_steps) if opt_config["decay_lr"] else opt_config["learning_rate"]
    for param_group in opt.param_groups:
        param_group["lr"] = lr

    seed = training_config["seed"] * (train_steps + 1) * WORLD_SIZE + RANK
    if train_steps % training_config["ckpt_every"] == 0:
        logger.info(f"Setting seed to {seed} at step {train_steps}")
        torch.manual_seed(seed)

    x1 = get_batch(train_steps, batch_size)
    x1 = x1.to(device=DEVICE, non_blocking=True)
    x0, x1 = generate_ot_pairs(x1, seed=seed)
    # x0 = torch.from_numpy(x0).to(device=DEVICE, non_blocking=True)
    # x0 = torch.randn_like(x1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        loss = compute_loss(model, x0, x1)
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), opt_config["max_grad_norm"]
    )
    opt.step()
    update_ema(ema, simple_model)

    running_loss += loss
    log_steps += 1
    train_steps += 1

    # Save DiT checkpoint:
    if train_steps % training_config["ckpt_every"] == 0 and train_steps > 0:
        if RANK == 0:
            checkpoint = {
                "model": simple_model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "config": CONFIG,
                "train_steps": train_steps,
                "batch_size": batch_size,
                "lr": lr,
            }
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            model.eval()
            img = sample(
                model,
                decoder,
                seed=0,
                input_size=model_config["input_size"],
                in_channels=model_config["in_channels"],
                device=DEVICE,
                n=25,
            )
            N, C, H, W = img.shape
            img = img.reshape(5, 5, 3, H, W)
            img = img.permute(0, 3, 1, 4, 2)
            img = img.reshape(5 * H, 5 * W, 3).float()
            img = img.clamp(0, 1)
            img = (img * 255).to(torch.uint8)
            # convert to PIL image
            img = Image.fromarray(img.data.cpu().numpy())
            img = img.resize((512, 512))
            model.train()
            wandb.log({"sample": wandb.Image(img)}, commit=False)
        if IS_DISTRIBUTED:
            dist.barrier()

    if train_steps % training_config["log_every"] == 0:
        # Measure training speed:
        torch.cuda.synchronize()
        end_time = time()
        steps_per_sec = log_steps / (end_time - start_time)
        # Reduce loss history over all processes:
        avg_loss = torch.tensor(running_loss.item() / log_steps, device=DEVICE)
        if IS_DISTRIBUTED:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / WORLD_SIZE
        logger.info(
            f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Grad Norm: {grad_norm:.4f}"
        )

        # Log to wandb
        if RANK == 0 and wandb_config["enable"]:
            # log img as images
            wandb.log(
                {
                    "avg_loss": avg_loss,
                    "steps_per_sec": steps_per_sec,
                    "step": train_steps,
                    "grad_norm": grad_norm,
                    "lr": lr,
                }
            )

        # Reset monitoring variables:
        running_loss = 0
        log_steps = 0
        start_time = time()


logger.info("Done!")

if RANK == 0:
    wandb.finish()

if IS_DISTRIBUTED:
    dist.destroy_process_group()
