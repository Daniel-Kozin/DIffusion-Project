"""
Quick check of how much memory a given batch size actually uses, without committing to
a full training run. Runs one real forward+backward pass and reports memory used.

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n uni python -m flow_model.mem_probe --batch_size 2
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n uni python -m flow_model.mem_probe --batch_size 4
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n uni python -m flow_model.mem_probe --batch_size 8
"""
import argparse

import torch

from .train import flow_matching_loss, get_device
from .velocity_model import FlowMatchingUNet3D


def probe(batch_size: int, base_ch: int, embed_channels: int, use_grad_checkpoint: bool,
          device_name: str) -> None:
    device = get_device(device_name)
    print(f"device={device}, batch_size={batch_size}, base_ch={base_ch}, "
          f"use_grad_checkpoint={use_grad_checkpoint}")

    if device.type == "mps":
        torch.mps.empty_cache()

    model = FlowMatchingUNet3D(base_ch=base_ch, embed_channels=embed_channels,
                                use_grad_checkpoint=use_grad_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    x1 = torch.rand(batch_size, 4, 26, 128, 128, device=device).round()  # fake one-hot-ish batch

    model.train()
    optimizer.zero_grad()
    loss = flow_matching_loss(model, x1, device)
    loss.backward()
    optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()
        current = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory() / 1e9
        print(f"MPS current_allocated_memory: {current:.2f} GB")
        print(f"MPS driver_allocated_memory:  {driver:.2f} GB  (includes MPS's own overhead/cache)")
    elif device.type == "cuda":
        current = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"CUDA memory_allocated: {current:.2f} GB")
        print(f"CUDA max_memory_allocated (peak): {peak:.2f} GB")
    else:
        print("CPU device — no GPU memory counter available; check Activity Monitor / `top` instead.")


def main():
    parser = argparse.ArgumentParser(description="Probe memory usage for a given batch size")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_ch", type=int, default=8)
    parser.add_argument("--embed_channels", type=int, default=16)
    parser.add_argument("--use_grad_checkpoint", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                         default=False)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    probe(args.batch_size, args.base_ch, args.embed_channels, args.use_grad_checkpoint, args.device)


if __name__ == "__main__":
    main()
