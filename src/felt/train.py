"""Train one model. Checkpoints resume at epoch boundaries, including RNG state."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from felt.data import load_config, make_dataset
from felt.models import CHANNELS, EnvelopeModel, loss


def environment():
    def git(*args):
        process = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return process.stdout.strip() if process.returncode == 0 else None

    source = Path(__file__).parent
    code = hashlib.sha256()
    for path in sorted(source.glob("*.py")):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    lock = Path("uv.lock")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("felt-research", "numpy", "torch", "matplotlib")
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_sha256": code.hexdigest(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
    }


def select_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this execution environment; use --device cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(name)


def train(config, out, mode="combined", resume=None, stop_after=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()) and resume is None:
        raise ValueError("Run directory is not empty. Resume it or choose a new directory.")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    device = select_device(config["device"])
    splits, manifest = make_dataset(config)
    tensors = {
        key: tuple(torch.from_numpy(a).to(device) for a in arrays)
        for key, arrays in splits.items()
        if key != "test"
    }
    model = EnvelopeModel(mode, config["hidden_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    start, best, history = 0, float("inf"), []
    best_model = None
    provenance = environment()
    if resume:
        # Our own checkpoint contains Python/NumPy RNG state. Never load untrusted .pt files.
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config or checkpoint["mode"] != mode:
            raise ValueError("Resume requires the same resolved config and input mode")
        if checkpoint["dataset_id"] != manifest["dataset_id"]:
            raise ValueError("Dataset changed since checkpoint")
        for key in ("source_sha256", "torch", "numpy", "python", "platform", "lock_sha256"):
            if checkpoint["environment"][key] != provenance[key]:
                raise ValueError(f"Resume environment mismatch: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Adam step counters remain on CPU; momentum buffers follow model parameters.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["rng_device"])
        elif device.type == "mps":
            torch.mps.set_rng_state(checkpoint["rng_device"])
        start, best, history = checkpoint["epoch"], checkpoint["best_val"], checkpoint["history"]
        best_model = checkpoint["best_model"]
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    x, y = tensors["train"]
    for epoch in range(start, min(config["epochs"], stop_after or config["epochs"])):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for batch in order.split(config["batch_size"]):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(x[batch]), y[batch])
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += value.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val = loss(model(tensors["val"][0]), tensors["val"][1]).item()
        scheduler.step()
        if val < best:
            best = val
            best_model = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        history.append({"epoch": epoch + 1, "train_loss": total / len(x), "val_loss": val})
        checkpoint = {
            "schema_version": 1,
            "epoch": epoch + 1,
            "next_epoch": epoch + 1,
            "config": config,
            "mode": mode,
            "dataset_id": manifest["dataset_id"],
            "environment": provenance,
            "model": model.state_dict(),
            "best_model": best_model,
            "best_val": best,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_device": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else torch.mps.get_rng_state()
            if device.type == "mps"
            else None,
        }
        temp = out / "last.tmp"
        torch.save(checkpoint, temp)
        temp.replace(out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)
    if best_model is not None:
        torch.save(
            {
                "state_dict": best_model,
                "mode": mode,
                "config": config,
                "dataset_id": manifest["dataset_id"],
                "environment": provenance,
            },
            out / "best.pt",
        )
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment001.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=CHANNELS, default="combined")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--resume")
    parser.add_argument(
        "--stop-after", type=int, help="Stop at an epoch boundary; keep full schedule"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    train(config, args.out, args.mode, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
