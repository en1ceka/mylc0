"""Inference network files -- the counterpart of Lc0's ``.pb.gz`` weights.

A network file is completely independent of the training setup: it carries the
model configuration, the weights, and enough metadata to identify which
generation produced it. The engine needs nothing else.

Layout (a ``torch.save`` archive, so it is self-describing and portable):

    magic           "MYLC0NET"
    format_version  1
    config          the ModelConfig as a JSON string
    metadata        generation, training step, input format, timestamps, ...
    state_dict      the model weights (float32)

``format_version`` is checked on load, so older generations keep loading after
the format evolves. Exported files are never overwritten by the training loop:
each generation gets its own ``gen_NNNNNN.mylc0`` plus a ``latest.mylc0`` copy.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from typing import Any, Dict, Optional, Tuple

import torch

from .config import ModelConfig, model_config_from_dict
from .model import LczeroModel, build_model

MAGIC = "MYLC0NET"
FORMAT_VERSION = 1
NETWORK_SUFFIX = ".mylc0"


def save_network(path: str, model: LczeroModel, config: ModelConfig,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_format": config.input_format,
        "policy_size": 1858,
        "parameters": model.num_parameters(),
    }
    if metadata:
        meta.update(metadata)
    state = {k: v.detach().to("cpu", torch.float32)
             for k, v in model.state_dict().items()}
    payload = {
        "magic": MAGIC,
        "format_version": FORMAT_VERSION,
        "config": json.dumps(dataclasses.asdict(config)),
        "metadata": json.dumps(meta),
        "state_dict": state,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_network(path: str, device: str = "cpu"
                 ) -> Tuple[LczeroModel, ModelConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("magic") != MAGIC:
        raise ValueError(f"{path} is not a mylc0 network file")
    version = int(payload.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(f"{path} has format version {version}, this build "
                         f"understands up to {FORMAT_VERSION}")
    config = model_config_from_dict(json.loads(payload["config"]))
    metadata = json.loads(payload.get("metadata", "{}"))
    model = build_model(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config, metadata


def read_metadata(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return json.loads(payload.get("metadata", "{}"))


def copy_as_latest(path: str) -> str:
    latest = os.path.join(os.path.dirname(path), "latest" + NETWORK_SUFFIX)
    shutil.copyfile(path, latest)
    return latest


def generation_filename(directory: str, generation: int) -> str:
    return os.path.join(directory, f"gen_{generation:06d}{NETWORK_SUFFIX}")


def export_onnx(path: str, model: LczeroModel, config: ModelConfig,
                batch_size: int = 1, opset: int = 17) -> str:
    """Optional ONNX export, mirroring Lc0's ``leela2onnx`` idea.

    Lc0 can carry an ONNX model inside its weight file (``OnnxModel`` in
    net.proto) and run it with the onnxruntime backend; this produces the same
    kind of artefact for anyone who prefers running the engine without PyTorch.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ONNX export needs the optional 'onnx' package "
                           "(pip install onnx)") from exc
    model = model.to("cpu").float().eval()
    dummy = torch.zeros(batch_size, 112, 8, 8)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner: LczeroModel, cfg: ModelConfig):
            super().__init__()
            self.inner = inner
            self.policy_head = cfg.primary_policy_head
            self.value_head = cfg.primary_value_head
            self.movesleft_head = cfg.primary_movesleft_head

        def forward(self, x):
            out = self.inner(x)
            mlh = (out.movesleft[self.movesleft_head]
                   if self.movesleft_head else torch.zeros(x.shape[0], 1))
            return (out.policy[self.policy_head],
                    out.value[self.value_head][0], mlh)

    torch.onnx.export(
        _Wrapper(model, config), dummy, path,
        input_names=["/input/planes"],
        output_names=["/output/policy", "/output/wdl", "/output/mlh"],
        dynamic_axes={"/input/planes": {0: "batch"},
                      "/output/policy": {0: "batch"},
                      "/output/wdl": {0: "batch"},
                      "/output/mlh": {0: "batch"}},
        opset_version=opset)
    return path
