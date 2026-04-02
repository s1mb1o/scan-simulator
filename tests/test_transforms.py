"""Smoke tests for all transforms and the pipeline."""

import numpy as np
import pytest

from scan_simulator import ScanSimulator
from scan_simulator.transforms import REGISTRY
from scan_simulator.transforms.base import TransformResult
from scan_simulator.presets import PRESETS


def _make_test_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a simple test floor plan image (white bg, black lines)."""
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    # Horizontal and vertical lines
    img[h // 4, w // 4:3 * w // 4] = 20
    img[3 * h // 4, w // 4:3 * w // 4] = 20
    img[h // 4:3 * h // 4, w // 4] = 20
    img[h // 4:3 * h // 4, 3 * w // 4] = 20
    # Cross line
    img[h // 2, w // 4:3 * w // 4] = 20
    return img


def _make_test_mask(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a matching test mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4, w // 4:3 * w // 4] = 255
    mask[3 * h // 4, w // 4:3 * w // 4] = 255
    mask[h // 4:3 * h // 4, w // 4] = 255
    mask[h // 4:3 * h // 4, 3 * w // 4] = 255
    mask[h // 2, w // 4:3 * w // 4] = 255
    return mask


class TestIndividualTransforms:
    """Test each transform individually."""

    @pytest.mark.parametrize("name", sorted(REGISTRY.keys()))
    def test_transform_runs(self, name: str):
        """Each transform should produce valid output."""
        img = _make_test_image()
        mask = _make_test_mask()
        rng = np.random.default_rng(42)

        transform = REGISTRY[name](rng=rng)
        result = transform(img, mask)

        assert isinstance(result, TransformResult)
        assert result.image.dtype == np.uint8
        assert result.image.shape[:2] == img.shape[:2]  # size preserved
        assert result.image.shape[2] == 3  # still BGR

    @pytest.mark.parametrize("name", sorted(REGISTRY.keys()))
    def test_transform_no_mask(self, name: str):
        """Each transform should work without a mask."""
        img = _make_test_image()
        rng = np.random.default_rng(42)

        transform = REGISTRY[name](rng=rng)
        result = transform(img, None)

        assert result.image.dtype == np.uint8
        assert result.mask is None

    @pytest.mark.parametrize("name", [n for n, t in REGISTRY.items()
                                       if t.affects_geometry])
    def test_geometric_transform_warps_mask(self, name: str):
        """Geometric transforms should modify the mask."""
        img = _make_test_image()
        mask = _make_test_mask()
        rng = np.random.default_rng(42)

        # Use strong params to ensure visible change
        params = {}
        if name == "Rotation":
            params = {"angle": 5.0}
        elif name == "Perspective":
            params = {"strength": 0.05}
        elif name == "FoldMark":
            params = {"count": 2, "shadow_width": 5}
        elif name == "Wrinkle":
            params = {"count": 3, "strength": 8.0}

        transform = REGISTRY[name](rng=rng, **params)
        result = transform(img, mask)

        assert result.mask is not None
        assert result.mask.dtype == np.uint8
        assert result.mask.shape == mask.shape


class TestPipeline:
    """Test the full pipeline."""

    @pytest.mark.parametrize("preset", sorted(PRESETS.keys()))
    def test_preset_runs(self, preset: str):
        """Each preset should produce valid output."""
        sim = ScanSimulator.from_preset(preset, seed=42)
        img = _make_test_image()
        mask = _make_test_mask()

        degraded, degraded_mask = sim(img, mask)

        assert degraded.dtype == np.uint8
        assert degraded.shape == img.shape
        assert degraded_mask is not None
        assert degraded_mask.shape == mask.shape

    def test_deterministic_with_seed(self):
        """Same seed should produce same output."""
        img = _make_test_image()
        sim1 = ScanSimulator.from_preset("scan-heavy", seed=123)
        sim2 = ScanSimulator.from_preset("scan-heavy", seed=123)

        out1, _ = sim1(img)
        out2, _ = sim2(img)

        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different output."""
        img = _make_test_image()
        sim1 = ScanSimulator.from_preset("scan-heavy", seed=1)
        sim2 = ScanSimulator.from_preset("scan-heavy", seed=2)

        out1, _ = sim1(img)
        out2, _ = sim2(img)

        assert not np.array_equal(out1, out2)

    def test_preview_grid(self):
        """Preview grid should produce correctly sized output."""
        sim = ScanSimulator.from_preset("scan-clean", seed=42)
        img = _make_test_image(200, 200)
        grid = sim.preview_grid(img, rows=2, cols=3)

        assert grid.shape[0] == 200 * 2
        assert grid.shape[1] == 200 * 3
        assert grid.shape[2] == 3
