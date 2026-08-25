"""Training loop, checkpointing and network export.

Corresponds to ``lczero_training/training/training.py`` plus its checkpoint and
export handling. One "generation" is ``steps_per_network`` optimizer steps,
after which a checkpoint is written and an inference network is exported --
that is upstream's ``ScheduleConfig.steps_per_network``.

A checkpoint holds everything needed to resume after the machine is switched
off: weights, optimizer state, step and generation counters, the model config,
the RNG state and the data-loader position counters.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch

from ..net.config import Config, ModelConfig, TrainingConfig
from ..progress import bar as _bar
from ..progress import format_eta as _format_eta
from ..net.model import LczeroModel, build_model
from ..net.netfile import copy_as_latest, generation_filename, save_network
from .dataset import TrainingDataLoader
from .losses import LczeroLoss, policy_accuracy
from .optim import build_optimizer, make_lr_schedule

log = logging.getLogger("mylc0.trainer")

CHECKPOINT_FORMAT_VERSION = 1


class Trainer:
    def __init__(self, config: Config, device: str = "cuda",
                 checkpoint_dir: Optional[str] = None,
                 networks_dir: Optional[str] = None,
                 tensorboard_dir: Optional[str] = None):
        self.config = config
        self.tcfg: TrainingConfig = config.training
        self.mcfg: ModelConfig = config.model
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.checkpoint_dir = checkpoint_dir or self.tcfg.checkpoint_path
        self.networks_dir = networks_dir or self.tcfg.networks_path
        self.tensorboard_dir = tensorboard_dir or self.tcfg.tensorboard_path

        self.model: LczeroModel = build_model(self.mcfg).to(self.device)
        if getattr(self.tcfg, "activation_checkpointing", False):
            self.model.encoders.grad_checkpointing = True
            log.info("activation checkpointing enabled for the encoder tower")
        self.lr_schedule = make_lr_schedule(self.tcfg.lr_schedule)
        self.optimizer = build_optimizer(self.model, self.tcfg.optimizer,
                                         self.lr_schedule(0))
        self.loss_fn = LczeroLoss(self.tcfg.losses)
        self.use_amp = (self.tcfg.mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.step = 0
        self.generation = 0
        self.positions_seen = 0
        self.writer = None
        self._device_reported = False
        self._make_writer()

    # -- device ------------------------------------------------------------
    def device_summary(self) -> str:
        """One line saying what the training will actually run on."""
        if self.device.type == "cuda":
            index = self.device.index or 0
            props = torch.cuda.get_device_properties(index)
            return (f"training on GPU: {props.name} "
                    f"(cuda:{index}, {props.total_memory / 1e9:.1f} GB, "
                    f"CUDA {torch.version.cuda}) | "
                    f"mixed precision fp16: {'on' if self.use_amp else 'off'} | "
                    f"activation checkpointing: "
                    f"{'on' if self.model.encoders.grad_checkpointing else 'off'} | "
                    f"batch {self.tcfg.batch_size} x "
                    f"{max(1, self.tcfg.gradient_accumulation)} accumulation")
        if torch.cuda.is_available():
            return (f"training on CPU -- but a CUDA device IS available "
                    f"({torch.cuda.get_device_name(0)}); pass --device cuda "
                    f"to use it")
        return "training on CPU (no CUDA device found; this will be very slow)"

    def report_device(self) -> None:
        """Log the device banner once per trainer."""
        if self._device_reported:
            return
        self._device_reported = True
        log.info("%s", self.device_summary())

    # -- progress ----------------------------------------------------------
    def _render_progress(self, progress, metrics: Dict[str, float], done: int,
                         total: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed > 0 else 0.0
        progress.set(
            f"training  [{_bar(done / max(1, total))}] {done}/{total}  "
            f"loss {metrics.get('total_loss', float('nan')):.3f}  "
            f"p {_first(metrics, 'policy/'):.2f} "
            f"v {_first(metrics, 'value/'):.2f} "
            f"m {_first(metrics, 'movesleft/'):.2f}  "
            f"acc {metrics.get('policy/accuracy', 0.0):.2f}  "
            f"{rate:.1f} it/s  ETA {_format_eta(done, total, elapsed)}")

    # -- tensorboard -------------------------------------------------------
    def _make_writer(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tensorboard_dir)
        except Exception as exc:  # tensorboard is optional
            log.warning("TensorBoard unavailable (%s); logging to console only", exc)
            self.writer = None

    def close(self) -> None:
        """Flush and release the TensorBoard writer."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None
                    ) -> None:
        step = self.step if step is None else step
        if self.writer is not None:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, step)

    # -- checkpoints -------------------------------------------------------
    def checkpoint_path(self, generation: int) -> str:
        return os.path.join(self.checkpoint_dir, f"generation_{generation:06d}",
                            "checkpoint.pt")

    def save_checkpoint(self) -> str:
        path = self.checkpoint_path(self.generation)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": self.step,
            "generation": self.generation,
            "positions_seen": self.positions_seen,
            "model_config": json.dumps(asdict(self.mcfg)),
            "training_config": json.dumps(asdict(self.tcfg)),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self._prune_checkpoints()
        log.info("checkpoint written: %s (step %d, generation %d)", path,
                 self.step, self.generation)
        return path

    def _prune_checkpoints(self) -> None:
        keep = self.tcfg.checkpoint_max_to_keep
        if keep <= 0 or not os.path.isdir(self.checkpoint_dir):
            return
        entries = sorted(d for d in os.listdir(self.checkpoint_dir)
                         if d.startswith("generation_"))
        for old in entries[:-keep]:
            shutil.rmtree(os.path.join(self.checkpoint_dir, old),
                          ignore_errors=True)

    def latest_checkpoint(self) -> Optional[str]:
        if not os.path.isdir(self.checkpoint_dir):
            return None
        entries = sorted(d for d in os.listdir(self.checkpoint_dir)
                         if d.startswith("generation_"))
        for name in reversed(entries):
            path = os.path.join(self.checkpoint_dir, name, "checkpoint.pt")
            if os.path.isfile(path):
                return path
        return None

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        path = path or self.latest_checkpoint()
        if path is None:
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        version = int(payload.get("format_version", 0))
        if version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"checkpoint {path} is from a newer format")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        # The scaler state is empty when the checkpoint was written without
        # mixed precision (for instance by init_network.py on CPU).
        if payload.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(payload["scaler"])
        self.step = int(payload["step"])
        self.generation = int(payload["generation"])
        self.positions_seen = int(payload.get("positions_seen", 0))
        try:
            torch.set_rng_state(payload["torch_rng"].cpu().to(torch.uint8))
            np.random.set_state(payload["numpy_rng"])
        except Exception:
            log.warning("could not restore RNG state from %s", path)
        log.info("resumed from %s at step %d, generation %d", path, self.step,
                 self.generation)
        return True

    # -- export ------------------------------------------------------------
    def export_network(self, generation: Optional[int] = None) -> str:
        generation = self.generation if generation is None else generation
        path = generation_filename(self.networks_dir, generation)
        save_network(path, self.model, self.mcfg, metadata={
            "generation": generation,
            "training_step": self.step,
            "positions_seen": self.positions_seen,
            "name": self.config.name,
        })
        copy_as_latest(path)
        log.info("network exported: %s", path)
        return path

    # -- training ----------------------------------------------------------
    def _upload(self, array: np.ndarray) -> torch.Tensor:
        """Host -> device through pinned memory, asynchronously.

        ``pin_memory()`` goes through PyTorch's caching host allocator, which
        keeps the staging buffer alive until the copy has actually run, so the
        non-blocking transfer is safe to overlap with the previous step.
        """
        tensor = torch.from_numpy(array)
        if self.device.type != "cuda":
            return tensor.to(self.device)
        return tensor.pin_memory().to(self.device, non_blocking=True)

    def train_step(self, loader: TrainingDataLoader) -> Dict[str, float]:
        self.model.train()
        accum = max(1, self.tcfg.gradient_accumulation)
        lr = self.lr_schedule(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        metrics_sum: Dict[str, float] = {}
        total_loss = 0.0
        seen = 0
        for _ in range(accum):
            planes, probs, values = _next_batch(loader)
            planes_t = self._upload(planes)
            probs_t = self._upload(probs)
            values_t = self._upload(values)
            seen += planes.shape[0]

            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                predictions = self.model(planes_t)
                loss, metrics = self.loss_fn(self.model, predictions, probs_t,
                                             values_t)
            self.scaler.scale(loss / accum).backward()
            total_loss += float(loss.detach())
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
            with torch.no_grad():
                acc = policy_accuracy(predictions, probs_t,
                                      self.mcfg.primary_policy_head)
            metrics_sum["policy/accuracy"] = (metrics_sum.get("policy/accuracy", 0.0)
                                              + float(acc))

        grad_norm = float("nan")
        if self.tcfg.max_grad_norm and self.tcfg.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tcfg.max_grad_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.step += 1
        self.positions_seen += seen
        out = {k: v / accum for k, v in metrics_sum.items()}
        out["total_loss"] = total_loss / accum
        out["lr"] = lr
        out["grad_norm"] = grad_norm
        return out

    def train_generation(self, loader: TrainingDataLoader,
                         steps: Optional[int] = None,
                         stop_check=None, progress=None) -> Dict[str, float]:
        """Run one generation's worth of steps, then checkpoint and export."""
        self.report_device()
        steps = steps or self.tcfg.steps_per_network
        start_step = self.step
        t0 = time.perf_counter()
        last: Dict[str, float] = {}
        if progress is not None:
            progress.set(f"training  [{_bar(0.0)}] 0/{steps}  "
                         f"waiting for the first batch", force=True)
        while self.step - start_step < steps:
            if stop_check is not None and stop_check():
                break
            last = self.train_step(loader)
            if progress is not None:
                self._render_progress(progress, last, self.step - start_step,
                                      steps, time.perf_counter() - t0)
            if self.step % self.tcfg.log_every_steps == 0:
                elapsed = time.perf_counter() - t0
                done = self.step - start_step
                pos_per_s = done * self.tcfg.batch_size * max(
                    1, self.tcfg.gradient_accumulation) / max(elapsed, 1e-9)
                scalars = {f"loss/{k}": v for k, v in last.items()
                           if k not in ("lr", "grad_norm")}
                scalars["train/lr"] = last["lr"]
                scalars["train/grad_norm"] = last["grad_norm"]
                scalars["train/positions_per_second"] = pos_per_s
                scalars["train/positions_seen"] = self.positions_seen
                scalars["data/chunks_read"] = loader.stats.chunks_read
                scalars["data/frames_sampled"] = loader.stats.frames_sampled
                scalars["data/chunk_pool"] = len(loader.pool)
                scalars["train/generation"] = self.generation
                scalars["train/step"] = self.step
                if self.device.type == "cuda":
                    scalars["gpu/memory_allocated_mb"] = (
                        torch.cuda.memory_allocated() / 1e6)
                    scalars["gpu/max_memory_allocated_mb"] = (
                        torch.cuda.max_memory_allocated() / 1e6)
                utilization = _gpu_utilization()
                if utilization is not None:
                    scalars["gpu/utilization_percent"] = utilization
                self.log_scalars(scalars)
                log.info(
                    "step %d gen %d | total %.4f | policy %.4f | value %.4f | "
                    "mlh %.4f | acc %.3f | lr %.2e | %.1f pos/s",
                    self.step, self.generation, last.get("total_loss", 0.0),
                    _first(last, "policy/"), _first(last, "value/"),
                    _first(last, "movesleft/"),
                    last.get("policy/accuracy", 0.0), last["lr"], pos_per_s)

        self.generation += 1
        self.save_checkpoint()
        self.export_network()
        return last


def _next_batch(loader, timeout: float = 600.0):
    """Next training batch, with a comprehensible error if data dries up."""
    try:
        return loader.next_batch(timeout=timeout)
    except TypeError:      # a fixed-batch stand-in without the timeout kwarg
        return loader.next_batch()
    except Exception as exc:
        raise RuntimeError(
            f"no training batch within {timeout:.0f}s -- is self-play still "
            f"producing chunks into the data directory?") from exc


def _gpu_utilization() -> Optional[float]:
    """GPU busy percentage, when the driver bindings are available."""
    try:
        return float(torch.cuda.utilization())
    except Exception:
        return None


def _first(metrics: Dict[str, float], prefix: str) -> float:
    for key, value in metrics.items():
        if key.startswith(prefix) and not key.endswith("accuracy"):
            return value
    return float("nan")
