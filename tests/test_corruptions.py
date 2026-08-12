import numpy as np
import pytest

from robustness.corruptions import CORRUPTIONS, gaussian_noise, brightness, fog, motion_blur


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(0)
    return rng.uniform(0.2, 0.8, size=(64, 64, 3)).astype(np.float32)


@pytest.mark.parametrize("name", list(CORRUPTIONS.keys()))
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_corruption_preserves_shape_and_range(sample_image, name, severity):
    out = CORRUPTIONS[name](sample_image, severity)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_gaussian_noise_severity_increases_deviation(sample_image):
    low = gaussian_noise(sample_image, 1)
    high = gaussian_noise(sample_image, 5)
    dev_low = np.abs(low - sample_image).mean()
    dev_high = np.abs(high - sample_image).mean()
    assert dev_high > dev_low


def test_brightness_increases_mean_intensity(sample_image):
    out = brightness(sample_image, 3)
    assert out.mean() > sample_image.mean()


def test_brightness_severity_monotonic(sample_image):
    low = brightness(sample_image, 1).mean()
    high = brightness(sample_image, 5).mean()
    assert high >= low


def test_fog_reduces_dynamic_range(sample_image):
    out = fog(sample_image, 5)
    assert out.std() < sample_image.std()


def test_motion_blur_reduces_local_variance(sample_image):
    out = motion_blur(sample_image, 5)
    assert np.abs(out - sample_image).mean() > 0


def test_all_four_required_corruption_types_present():
    assert set(CORRUPTIONS.keys()) == {"gaussian_noise", "motion_blur", "brightness", "fog"}
