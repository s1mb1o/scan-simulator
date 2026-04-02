# scan-simulator — Visual Demo

All examples below use a single clean digital floor plan as input, processed
through each individual transform (with strong parameters for visibility)
and through each preset pipeline (6 random variants per preset).

## Overview

All 25 transforms at a glance — original on the left in each row, transforms to the right:

![All transforms overview](docs/images/composite_all_transforms.png)

---

## Individual Transforms

### Paper & Aging

**PaperColor** — Non-uniform yellowing/sepia tint with smooth spatial variation.

![PaperColor](docs/images/PaperColor.png)

**PaperTexture** — Procedural grain and fiber pattern overlay.

![PaperTexture](docs/images/PaperTexture.png)

**InkFading** — Spatially varying contrast reduction on dark pixels (ink lines fade unevenly).

![InkFading](docs/images/InkFading.png)

**CoffeeStain** — Elliptical brownish blotches with characteristic ring edges.

![CoffeeStain](docs/images/CoffeeStain.png)

**Foxing** — Scattered small brown spots simulating aged paper fungal damage.

![Foxing](docs/images/Foxing.png)

---

### Physical Damage

**FoldMark** — Straight crease lines with shadow and slight geometric offset.

![FoldMark](docs/images/FoldMark.png)

**Wrinkle** — Local elastic deformation with highlight/shadow gradients.

![Wrinkle](docs/images/Wrinkle.png)

**EdgeWear** — Organic edge wear with wavy spline boundaries, darkening, and scattered noise.

![EdgeWear](docs/images/EdgeWear.png)

**SurfaceWear** — Scratches, scuff marks, and faded bands from handling and abrasion.

![SurfaceWear](docs/images/SurfaceWear.png)

**Holes** — Punch holes (clean circles with shadow ring), worm holes (tiny irregular), and torn spots (jagged outline with fiber edges).

![Holes](docs/images/Holes.png)

---

### Scanner Artifacts

**Rotation** — Slight misalignment as if paper was placed crooked in scanner.

![Rotation](docs/images/Rotation.png)

**Perspective** — Mild trapezoid distortion (photographed at an angle).

![Perspective](docs/images/Perspective.png)

**UnevenLight** — Vignetting, directional gradient, or flash hotspot.

![UnevenLight](docs/images/UnevenLight.png)

**ScannerShadow** — Dark gradient along one edge (scanner lid shadow).

![ScannerShadow](docs/images/ScannerShadow.png)

**Moire** — Sinusoidal interference pattern from scanning printed halftones.

![Moire](docs/images/Moire.png)

**DirtyRollers** — Horizontal/vertical banding from contaminated scanner rollers with non-uniform pressure.

![DirtyRollers](docs/images/DirtyRollers.png)

**BookBinding** — Page curvature and shadow near book spine, with slight horizontal compression.

![BookBinding](docs/images/BookBinding.png)

---

### Camera Artifacts

**DefocusBlur** — Spatially varying blur: sharp center, soft edges (depth-of-field).

![DefocusBlur](docs/images/DefocusBlur.png)

**MotionBlur** — Directional smear from hand shake.

![MotionBlur](docs/images/MotionBlur.png)

**GaussianNoise** — Sensor noise (Gaussian + salt-and-pepper).

![GaussianNoise](docs/images/GaussianNoise.png)

**JPEGArtifacts** — Block compression artifacts at low quality.

![JPEGArtifacts](docs/images/JPEGArtifacts.png)

**ChromaticAberration** — Color fringing at high-contrast edges (R/B channel shift).

![ChromaticAberration](docs/images/ChromaticAberration.png)

---

### Drawing Artifacts

**PenBleed** — Ink bleeding: dark lines thicken with fuzzy edges.

![PenBleed](docs/images/PenBleed.png)

**HandAnnotation** — Random pen marks and scribbles in margin areas.

![HandAnnotation](docs/images/HandAnnotation.png)

**Watermark** — Semi-transparent diagonal text overlay.

![Watermark](docs/images/Watermark.png)

---

## Presets

Each preset combines multiple transforms with tuned probability and severity ranges.
Shown as 2x3 grids of random variants from the same input image.

### scan-clean

Well-maintained office scanner. Light degradation — subtle noise, slight rotation, minor color shift.

![scan-clean preset](docs/images/scan_clean.png)

### scan-heavy

Old or cheap scanner with aged paper. Paper yellowing, fold marks, shadows, noise, compression.

![scan-heavy preset](docs/images/scan_heavy.png)

### photo-indoor

Phone photo under indoor lighting. Perspective distortion, uneven illumination, blur, noise.

![photo-indoor preset](docs/images/photo_indoor.png)

### photo-outdoor

Phone photo in natural light. Lighter version with less perspective and better illumination.

![photo-outdoor preset](docs/images/photo_outdoor.png)

### photocopy

Multi-generation photocopy. Ink bleed, fading, moire patterns, paper texture.

![photocopy preset](docs/images/photocopy.png)

### archive

Aged/stored document. Heavy yellowing, foxing spots, coffee stains, fold marks, edge wear.

![archive preset](docs/images/archive.png)
