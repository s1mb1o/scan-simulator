# scan-simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/scan-simulator.svg)](https://pypi.org/project/scan-simulator/)

Synthetic degradation pipeline that transforms clean digital images into
realistic **scans**, **phone photos**, **photocopies**, and **aged documents**.
Built for **data augmentation** in document AI, OCR, layout analysis, and
floor plan recognition pipelines.

![All transforms overview](docs/images/composite_all_transforms.png)

**[See visual demo of all 25 transforms and 6 presets](DEMO.md)**

## Key Features

- **25 physically-motivated transforms** — paper aging, fold marks, wrinkles,
  scanner shadows, moiré, lens blur, JPEG artifacts, chromatic aberration, and more
- **6 ready-to-use presets** — from clean office scans to heavily degraded photocopies
- **Mask-aware** — applies identical geometric transforms to ground-truth masks,
  keeping segmentation labels aligned
- **CLI + Python API** — use as a command-line tool or integrate into training loops
- **Batch processing** — parallel processing of entire directories
- **Configurable** — YAML-based configuration, combine and tune any transforms
- **Deterministic seeds** — reproducible augmentation for experiment tracking

## Installation

```bash
pip install scan-simulator
```

Or install from source:

```bash
git clone https://github.com/s1mb1o/scan-simulator.git
cd scan-simulator
pip install -e .
```

## Quick Start

### Command Line

```bash
# Single image with default preset
scan-simulator input.png -o output.png

# Batch directory (parallel)
scan-simulator input_dir/ -o output_dir/ --workers 8

# Specific preset
scan-simulator input.png -o output.png --preset photocopy

# With ground-truth mask (geometric transforms applied to both)
scan-simulator input.png -o output.png --mask mask.png --mask-out mask_out.png

# Preview grid (3x3 random variants)
scan-simulator input.png --preview 3x3 -o grid.png
```

### Python API

```python
from scan_simulator import ScanSimulator

sim = ScanSimulator.from_preset("scan-heavy")
degraded_image, degraded_mask = sim(image, mask=mask)
```

## Presets

| Preset | Simulates | Severity |
|--------|-----------|----------|
| `scan-clean` | Well-maintained office scanner | Light |
| `scan-heavy` | Old/cheap scanner, aged paper | Heavy |
| `photo-indoor` | Phone photo under indoor lighting | Medium |
| `photo-outdoor` | Phone photo in natural light | Light-Medium |
| `photocopy` | Multi-generation photocopy | Heavy |
| `archive` | Aged/stored document (yellowed, foxed) | Medium-Heavy |

## Transforms

### Paper & Aging
- **Paper color** — non-uniform yellowing, coffee-stain patches, foxing spots
- **Paper texture** — visible grain/fiber pattern overlaid on the image
- **Ink fading** — partial loss of line contrast, especially thin lines

### Physical Damage
- **Fold marks** — straight crease lines with slight offset and shadow
- **Wrinkles** — local elastic deformation with highlight/shadow
- **Edge wear** — organic wavy-boundary margin darkening with speckle noise
- **Surface wear** — scratches, scuff marks, and faded bands from handling
- **Holes** — punch holes, worm/insect holes, torn spots with fiber edges

### Scanner Artifacts
- **Rotation** — slight misalignment (+-5 deg) with background fill
- **Perspective** — mild trapezoid distortion (as if photographed at angle)
- **Uneven illumination** — vignetting, light falloff at edges, flash hotspot
- **Scanner lid shadow** — dark gradient along one edge
- **Moire pattern** — interference pattern from scanning printed halftones
- **Dirty rollers** — horizontal/vertical banding from contaminated scanner rollers
- **Book binding** — page curvature and darkening near spine fold

### Camera/Photo Artifacts
- **Lens blur** — slight defocus, especially at edges (depth-of-field)
- **Motion blur** — directional smear from hand shake
- **Noise** — Gaussian + salt-and-pepper (sensor noise)
- **JPEG compression** — block artifacts at various quality levels
- **Chromatic aberration** — color fringing at high-contrast edges

### Drawing Artifacts
- **Hand annotations** — random scribbles/marks in margins
- **Stamps/watermarks** — semi-transparent overlay
- **Pen bleed** — line thickening with fuzzy edges

## Use Cases

- **Document AI / OCR training** — bridge the domain gap between clean digital
  inputs and real-world scanned documents
- **Floor plan recognition** — augment clean CAD exports to match scanned blueprints
- **Layout analysis** — train models robust to scan quality variations
- **Historical document processing** — simulate aging and degradation artifacts
- **Quality assurance** — stress-test document processing pipelines

## Dependencies

- Python 3.10+
- OpenCV
- NumPy
- Pillow
- SciPy
- scikit-image
- PyYAML

## License

[MIT](LICENSE)
