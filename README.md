# scan-simulator

Synthetic degradation pipeline that transforms clean digital images (floor plans,
technical drawings, documents) into realistic scan/photo-like images. Useful for
training and evaluating CV models on inputs that look like real-world scans, phone
photos, and photocopies.

**[See visual demo of all 21 transforms and 6 presets](DEMO.md)**

## Why

CV models trained on clean digital images often fail on real-world inputs: scanned
PDFs, phone photos of blueprints, photocopies, aged drawings. This domain gap can
be bridged by synthetically degrading clean training images with physically-motivated
transforms that simulate real scanning/photography artifacts — without collecting
thousands of real scans.

## Degradation Effects

### Paper & Aging
- **Paper color** — non-uniform yellowing, coffee-stain patches, foxing spots
- **Paper texture** — visible grain/fiber pattern overlaid on the image
- **Ink fading** — partial loss of line contrast, especially thin lines

### Physical Damage
- **Fold marks** — straight crease lines with slight offset and shadow
- **Wrinkles** — local elastic deformation with highlight/shadow
- **Edge wear** — darkened/damaged margins, dog-ears

### Scanner Artifacts
- **Rotation** — slight misalignment (±5°) with background fill
- **Perspective** — mild trapezoid distortion (as if photographed at angle)
- **Uneven illumination** — vignetting, light falloff at edges, flash hotspot
- **Scanner lid shadow** — dark gradient along one edge
- **Moiré pattern** — interference pattern from scanning printed halftones

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

## Usage

```bash
# Single image
python -m scan_simulator input.png -o output.png

# Batch directory (parallel)
python -m scan_simulator input_dir/ -o output_dir/ --workers 8

# With specific preset
python -m scan_simulator input.png -o output.png --preset scan-clean
python -m scan_simulator input.png -o output.png --preset scan-heavy
python -m scan_simulator input.png -o output.png --preset photo-indoor
python -m scan_simulator input.png -o output.png --preset photocopy

# With GT mask (applies same geometric transforms to mask)
python -m scan_simulator input.png -o output.png --mask mask.png --mask-out mask_out.png

# Preview grid (multiple random variants of one image)
python -m scan_simulator input.png --preview 3x3 -o grid.png

# Custom config
python -m scan_simulator input.png -o output.png --config custom.yaml
```

## Integration with Training

```python
from scan_simulator import ScanSimulator

sim = ScanSimulator.from_preset("scan-heavy")
degraded_image, degraded_mask = sim(image, mask=mask)
```

## Presets

| Preset | Use Case | Severity |
|--------|----------|----------|
| `scan-clean` | Well-maintained office scanner | Light |
| `scan-heavy` | Old/cheap scanner, aged paper | Heavy |
| `photo-indoor` | Phone photo under indoor lighting | Medium |
| `photo-outdoor` | Phone photo in natural light | Light–Medium |
| `photocopy` | Multi-generation photocopy | Heavy |
| `archive` | Aged/stored document (yellowed, foxed) | Medium–Heavy |

## Dependencies

- opencv-python
- numpy
- Pillow
- scipy (for elastic deformation)
- scikit-image (for noise models)
