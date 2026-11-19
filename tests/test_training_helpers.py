import numpy as np
from PIL import Image

from waste_ai.training import read_image_pil, training_class_weights


def test_preprocessing_preserves_aspect_ratio_with_letterbox(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (40, 20), (220, 10, 10)).save(path)
    image = read_image_pil(path, width=40, height=40)
    assert image.shape == (40, 40, 3)
    assert np.all(image[0] == 0)
    assert tuple(image[20, 20].astype(int)) == (220, 10, 10)


def test_balanced_weights_use_supplied_training_counts():
    weights = training_class_weights([0] * 6 + [1] * 2)
    assert weights[0] == 2 / 3
    assert weights[1] == 2.0


def test_balanced_weights_require_both_classes():
    import pytest
    with pytest.raises(ValueError, match="both classes"):
        training_class_weights([1, 1, 1])
