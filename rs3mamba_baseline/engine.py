"""Tensor-logit training and evaluation with sample-correct gradient accumulation."""
from __future__ import annotations

import logging
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import confusion_matrix, metrics_from_confusion


def resized_logits(model, images, size):
    logits = model(images)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Model must return Bx2xHxW tensor logits.")
    return logits if logits.shape[-2:] == tuple(size) else F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, log_every=20,
                accumulation_steps=1, scaler=None, amp=False, max_batches=None):
    logger = logging.getLogger("rs3mamba")
    model.train()
    started = time.monotonic()
    total_loss, num_images, group_images, steps = 0.0, 0, 0, 0
    count = min(len(loader), max_batches) if max_batches else len(loader)
    if count < 1:
        raise ValueError("Empty training loader.")
    scaler = scaler or torch.cuda.amp.GradScaler(enabled=False)
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, total=count, desc=f"Train {epoch}/{total_epochs}", disable=not sys.stderr.isatty())
    for step, batch in enumerate(bar, 1):
        if step > count:
            break
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            loss = criterion(resized_logits(model, images, masks.shape[-2:]), masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {step}.")
        # Sum sample losses now; divide gradients by the ACTUAL sample count at update time.
        # This also treats a short final batch/group correctly.
        batch_size = images.shape[0]
        scaler.scale(loss * batch_size).backward()
        group_images += batch_size
        total_loss += loss.item() * batch_size
        num_images += batch_size
        if step % accumulation_steps == 0 or step == count:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_images)
                    if not amp and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}, batch {step}.")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_images = 0
            steps += 1
        if step == 1 or step % log_every == 0 or step == count:
            logger.info("Train %s/%s batch %s/%s | loss %.6f | %.1fs", epoch, total_epochs,
                        step, count, total_loss / num_images, time.monotonic() - started)
        bar.set_postfix(loss=f"{total_loss / num_images:.5f}")
    return dict(loss=total_loss / num_images, seconds=time.monotonic() - started,
                num_images=num_images, optimizer_steps=steps)


@torch.inference_mode()
def evaluate_model(model, loader, device, criterion=None, description="Evaluate", log_every=20,
                   collect_per_image=True, sample_limit=0, max_batches=None, prediction_callback=None):
    logger = logging.getLogger("rs3mamba")
    model.eval()
    started = time.monotonic()
    matrix = np.zeros((2, 2), dtype=np.int64)
    per_image, samples = [], []
    total_loss, num_images = 0.0, 0
    count = min(len(loader), max_batches) if max_batches else len(loader)
    if count < 1:
        raise ValueError("Empty evaluation loader.")
    bar = tqdm(loader, total=count, desc=description, disable=not sys.stderr.isatty())
    for step, batch in enumerate(bar, 1):
        if step > count:
            break
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        logits = resized_logits(model, images, masks.shape[-2:])
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Nonfinite logits in {description}, batch {step}.")
        if criterion is not None:
            loss = criterion(logits, masks).item()
            if not math.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {description}, batch {step}.")
            total_loss += loss * images.size(0)
        predictions, targets = logits.argmax(1).cpu().numpy().astype(np.uint8), masks.cpu().numpy()
        for index, name in enumerate(batch["name"]):
            cm = confusion_matrix(targets[index], predictions[index])
            matrix += cm
            if collect_per_image:
                per_image.append(dict(name=name, **metrics_from_confusion(cm)))
            sample = dict(name=name, image_path=batch["image_path"][index],
                          prediction=predictions[index], target=targets[index])
            if prediction_callback:
                prediction_callback(sample)
            if len(samples) < sample_limit:
                samples.append(sample)
        num_images += images.size(0)
        if step == 1 or step % log_every == 0 or step == count:
            logger.info("%s batch %s/%s | %s images | %.1fs", description, step, count, num_images, time.monotonic() - started)
    return dict(metrics=metrics_from_confusion(matrix), confusion=matrix, per_image=per_image,
                samples=samples, num_images=num_images, loss=total_loss / num_images if criterion else None,
                seconds=time.monotonic() - started)
