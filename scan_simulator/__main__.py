"""CLI entry point for scan-simulator."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def process_single(
    input_path: str,
    output_path: str,
    mask_path: str | None,
    mask_out_path: str | None,
    preset: str | None,
    config_path: str | None,
    seed: int | None,
) -> str:
    """Process a single image (runs in worker process)."""
    from .pipeline import ScanSimulator

    if config_path:
        sim = ScanSimulator.from_yaml(config_path, seed=seed)
    elif preset:
        sim = ScanSimulator.from_preset(preset, seed=seed)
    else:
        sim = ScanSimulator.from_preset("scan-heavy", seed=seed)

    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        return f"SKIP {input_path}: cannot read"

    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    degraded, degraded_mask = sim(image, mask)
    cv2.imwrite(output_path, degraded)

    if degraded_mask is not None and mask_out_path:
        cv2.imwrite(mask_out_path, degraded_mask)

    return f"OK {output_path}"


def main():
    parser = argparse.ArgumentParser(
        prog="scan-simulator",
        description="Synthetic scan/photo degradation for document images",
    )
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", required=True,
                        help="Output image or directory")
    parser.add_argument("--preset", default=None,
                        help="Preset name (scan-clean, scan-heavy, photo-indoor, "
                             "photo-outdoor, photocopy, archive)")
    parser.add_argument("--config", default=None,
                        help="Path to custom YAML config")
    parser.add_argument("--mask", default=None,
                        help="Input GT mask (single image mode)")
    parser.add_argument("--mask-out", default=None,
                        help="Output GT mask path (single image mode)")
    parser.add_argument("--mask-dir", default=None,
                        help="Input mask directory (batch mode)")
    parser.add_argument("--mask-out-dir", default=None,
                        help="Output mask directory (batch mode)")
    parser.add_argument("--preview", default=None, metavar="RxC",
                        help="Generate preview grid (e.g., 3x3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (batch mode)")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available presets and exit")

    args = parser.parse_args()

    if args.list_presets:
        from .presets import PRESETS
        for name in sorted(PRESETS.keys()):
            n = len(PRESETS[name]["transforms"])
            print(f"  {name:20s} ({n} transforms)")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Preview mode
    if args.preview:
        from .pipeline import ScanSimulator
        rows, cols = map(int, args.preview.lower().split("x"))
        if args.config:
            sim = ScanSimulator.from_yaml(args.config, seed=args.seed)
        elif args.preset:
            sim = ScanSimulator.from_preset(args.preset, seed=args.seed)
        else:
            sim = ScanSimulator.from_preset("scan-heavy", seed=args.seed)

        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: cannot read {input_path}", file=sys.stderr)
            sys.exit(1)

        mask = None
        if args.mask:
            mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

        grid = sim.preview_grid(image, rows=rows, cols=cols, mask=mask)
        cv2.imwrite(str(output_path), grid)
        print(f"Preview grid ({rows}x{cols}) saved to {output_path}")
        return

    # Single image mode
    if input_path.is_file():
        result = process_single(
            str(input_path), str(output_path),
            args.mask, args.mask_out,
            args.preset, args.config, args.seed,
        )
        print(result)
        return

    # Batch directory mode
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    if args.mask_out_dir:
        Path(args.mask_out_dir).mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = sorted(p for p in input_path.iterdir()
                   if p.suffix.lower() in image_exts)

    if not files:
        print(f"No images found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} images with {args.workers} worker(s)...")
    t0 = time.time()

    tasks = []
    for f in files:
        out_f = str(output_path / f.name)
        mask_f = str(Path(args.mask_dir) / f.name) if args.mask_dir else None
        mask_out_f = (str(Path(args.mask_out_dir) / f.name)
                      if args.mask_out_dir else None)
        # Per-file seed for reproducibility (derived from global seed + filename)
        file_seed = None
        if args.seed is not None:
            file_seed = hash((args.seed, f.name)) % (2**31)
        tasks.append((str(f), out_f, mask_f, mask_out_f,
                       args.preset, args.config, file_seed))

    done = 0
    if args.workers <= 1:
        for task_args in tasks:
            result = process_single(*task_args)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(files)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_single, *t) for t in tasks]
            for fut in as_completed(futures):
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(files)}")

    elapsed = time.time() - t0
    print(f"Done: {len(files)} images in {elapsed:.1f}s "
          f"({elapsed / len(files) * 1000:.0f} ms/image)")


if __name__ == "__main__":
    main()
