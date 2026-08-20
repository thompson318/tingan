import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from timellm.data_provider.data_factory import data_provider
from timellm.models import TimeLLM
from timellm.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    vali_pulsar,
)
from tqdm import tqdm

from tingan.datasets import partim_to_timellm_format, split_tim_and_par_files
from tingan.networks import TimeSeriesDiscriminator, trainable_parameters
from tingan.plots import (
    plot_labels,
    plot_losses,
    plot_timellm_residuals,
    plot_timing_noise,
)

# Setting some environment variables and random seed, from Time-LLM original scripts
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 2026
random.seed(fix_seed)
torch.manual_seed(fix_seed)

# Loading configuration
with Path("timellm_config.json").open() as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
parser = argparse.ArgumentParser()
args = parser.parse_args(namespace=t_args)

# Checking configuration
if args.model != "TimeLLM":
    model_err_msg = "Model should be TimeLLM."
    raise ValueError(model_err_msg)

if args.use_amp:
    use_amp_err_msg = "use_amp should be False."
    raise ValueError(use_amp_err_msg)

if len(args.d_updates_per_batch) != len(args.d_updates_epochs):
    len_err_msg = (
        "d_updates_per_batch and d_updates_epochs should have the same length."
    )
    raise ValueError(len_err_msg)

if args.d_updates_epochs[0] != 0:
    d_err_msg = "d_updates_epochs should start from 0."
    raise ValueError(d_err_msg)

# Setting up distributed training and accelerator, from Time-LLM original scripts
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# Setting record of experiments
setting = (
    f"{args.task_name}_"
    f"{args.model_id}_"
    f"{args.model}_"
    f"{args.data}_"
    f"ft{args.features}_"
    f"sl{args.seq_len}_"
    f"ll{args.label_len}_"
    f"pl{args.pred_len}_"
    f"dm{args.d_model}_"
    f"nh{args.n_heads}_"
    f"el{args.e_layers}_"
    f"dl{args.d_layers}_"
    f"df{args.d_ff}_"
    f"fc{args.factor}_"
    f"eb{args.embed}_"
    f"{args.des}"
)

path_data = Path(args.root_path) / Path(args.data_path)
if not path_data.exists():
    n = split_tim_and_par_files(
        path_data.with_suffix(".tim"), path_data.with_suffix(".par")
    )
    [*_, prefix, _] = args.data_path.split(".")
    dfs = []
    for i in range(n):
        j = i if i < 10 else i + 1
        dfs.append(
            partim_to_timellm_format(
                Path(args.root_path) / Path(f"{prefix}_{j}").with_suffix(".par"),
                Path(args.root_path) / Path(f"{prefix}_{i}").with_suffix(".tim"),
            )
        )
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_csv(path_data, header=["date", "resid_s", "err_s"], index=False)

# Creating training, validation and test datasets
train_data, train_loader = data_provider(args, "train")
vali_data, vali_loader = data_provider(args, "val")
test_data, test_loader = data_provider(args, "test")

# Creating generator and discriminator
model = TimeLLM.Model(args).float()
discriminator = TimeSeriesDiscriminator(
    seq_len=args.pred_len, n_channels=1 if args.features == "S" else 0
)

# Labels
real_label = torch.full((1,), 1.0, device=accelerator.device)
fake_label = torch.full((1,), 0.0, device=accelerator.device)
bce_loss = torch.nn.BCELoss()

# Creating directory where results will be saved
path = Path(args.checkpoints) / Path(setting + "-" + args.model_comment)
if not path.exists() and accelerator.is_local_main_process:
    path.mkdir(parents=True)

with (path / Path("timellm_config.json")).open("w") as f:
    args_dict = vars(args)
    json.dump(args_dict, f, indent=4)

args.d_updates_per_batch = args.d_updates_per_batch[::-1]
args.d_updates_epochs = np.array(args.d_updates_epochs).astype(int)

fig_timellm_residuals = plot_timellm_residuals(path_data)
fig_timellm_residuals.savefig(path / Path("residuals.png"))

train_steps = len(train_loader)
early_stopping = EarlyStopping(
    accelerator=accelerator, patience=args.patience, verbose=True
)

# Optimizers and schedulter
model_optim = torch.optim.Adam(trainable_parameters(model), lr=args.learning_rate)
discr_optim = torch.optim.Adam(
    trainable_parameters(discriminator), lr=1e-4, betas=(0.5, 0.999)
)

if args.lradj == "COS":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optim, T_max=20, eta_min=1e-8
    )
else:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate,
    )

model, model_optim, scheduler = accelerator.prepare(model, model_optim, scheduler)
discriminator, discr_optim = accelerator.prepare(discriminator, discr_optim)

train_loss_g = []
train_loss_d = []
dlabels_for_real = []
dlabels_for_mock = []
vali_loss_d = []

d_updates_per_batch = 1

for epoch in range(args.train_epochs):
    if epoch in args.d_updates_epochs:
        d_updates_per_batch = args.d_updates_per_batch.pop()

    model.train()
    discriminator.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
        enumerate(train_loader)
    ):
        # decoder input
        dec_inp = (
            torch.zeros_like(
                batch_y[:, -args.pred_len :, :].float().to(accelerator.device)
            )
            .float()
            .to(accelerator.device)
        )
        dec_inp = (
            torch.cat(
                [
                    batch_y[:, : args.label_len, :].float().to(accelerator.device),
                    dec_inp,
                ],
                dim=1,
            )
            .float()
            .to(accelerator.device)
        )

        # encoder - decoder
        if args.output_attention:
            outputs = model(
                batch_x.float().to(accelerator.device),
                batch_x_mark.float().to(accelerator.device),
                dec_inp,
                batch_y_mark.float().to(accelerator.device),
            )[0]
        else:
            outputs = model(
                batch_x.float().to(accelerator.device),
                batch_x_mark.float().to(accelerator.device),
                dec_inp,
                batch_y_mark.float().to(accelerator.device),
            )

        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len :, f_dim:]
        batch_y_pred = (
            batch_y[:, -args.pred_len :, f_dim:].float().to(accelerator.device)
        )

        # =========================================================
        #  TRAIN DISCRIMINATOR
        # =========================================================
        for idiscr in range(d_updates_per_batch):
            discr_optim.zero_grad()

            # Real samples
            real_data = batch_y_pred.detach()  # (batch, pred_len, n_channels)
            d_real = discriminator(real_data)  # (batch, 1l
            # Expand labels to match batch
            labels_real = real_label.expand_as(d_real)
            loss_d_real = bce_loss(d_real, labels_real)
            dlabels_for_real.append(d_real.detach().mean().item())
            accelerator.backward(loss_d_real)

            # Fake samples
            fake_data = outputs.detach()  # detach so gradient doesn't flow to generator
            d_fake = discriminator(fake_data)
            labels_fake = fake_label.expand_as(d_fake)
            loss_d_fake = bce_loss(d_fake, labels_fake)
            dlabels_for_mock.append(d_fake.detach().mean().item())
            accelerator.backward(loss_d_fake)

            if i == 0 and idiscr == d_updates_per_batch - 1:
                fig, ax = plot_timing_noise(
                    None,
                    batch_y_pred[:, :, 0].cpu().detach().numpy(),
                    labels=d_real.cpu().detach().numpy().round(3),
                )
                fig, ax = plot_timing_noise(
                    None,
                    outputs[:, :, 0].cpu().detach().numpy(),
                    labels=d_fake.cpu().detach().numpy().round(3),
                    fig=fig,
                    ax=ax,
                )
                fig.savefig(path / Path(f"outputs_epoch{epoch}_i{i}.pdf"))

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discr_optim.step()

            train_loss_d.append((loss_d_real + loss_d_fake).item())

        # =========================================================
        #  TRAIN GENERATOR (Time-LLM) — MSE + Adversarial
        # =========================================================
        model_optim.zero_grad()

        # Adversarial: we want the discriminator to think forecasts are REAL
        d_fake_for_g = discriminator(
            outputs
        )  # NO detach here — gradient flows to generator
        labels_for_g = real_label.expand_as(
            d_fake_for_g
        )  # generator wants "real" verdict
        loss_adv = bce_loss(d_fake_for_g, labels_for_g)

        loss_g = loss_adv
        train_loss_g.append(loss_g.item())

        accelerator.backward(loss_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model_optim.step()

        if args.lradj == "TST":
            adjust_learning_rate(
                accelerator, model_optim, scheduler, epoch + 1, args, printout=False
            )
            scheduler.step()

    accelerator.print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
    vali_loss, vali_loss_d, vali_pred_lab, vali_true_lab = vali_pulsar(
        args, accelerator, model, discriminator, vali_data, vali_loader, bce_loss
    )
    train_loss_g.append(np.nan)
    train_loss_d.append(np.nan)
    dlabels_for_real.append(np.nan)
    dlabels_for_mock.append(np.nan)
    accelerator.print(
        f"Epoch: {epoch + 1} | Train Loss: {train_loss_g[-1]:.7f} "
        f"Train Loss D: {np.mean(train_loss_d[-d_updates_per_batch:]):.7f} "
        f"Test Loss: {vali_loss:.7f} "
        f"Test Loss D: {vali_loss_d:.7f}"
    )
    accelerator.print(
        f"\titers: {i + 1}, epoch: {epoch + 1} | "
        f"D_loss_real: {loss_d_real.item():.7f} | "
        f"D_loss_fake: {loss_d_fake.item():.7f} | "
        f"G_adv: {loss_adv.item():.7f}"
    )

    early_stopping(vali_loss, model, str(path), discriminator)
    if early_stopping.early_stop:
        accelerator.print("Early stopping")
        break

    if args.lradj != "TST":
        if args.lradj == "COS":
            scheduler.step()
            accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
        else:
            if epoch == 0:
                args.learning_rate = model_optim.param_groups[0]["lr"]
                accelerator.print(
                    "lr = {:.10f}".format(model_optim.param_groups[0]["lr"])
                )
            adjust_learning_rate(
                accelerator, model_optim, scheduler, epoch + 1, args, printout=True
            )

    else:
        accelerator.print(f"Updating learning rate to {scheduler.get_last_lr()[0]}")

accelerator.wait_for_everyone()

fig = plot_losses(train_loss_g, train_loss_d)
fig.savefig(path / Path("loss.pdf"))
fig = plot_labels(dlabels_for_real, dlabels_for_mock)
fig.savefig(path / Path("labels.pdf"))
