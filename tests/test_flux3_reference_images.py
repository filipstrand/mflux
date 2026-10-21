"""FLUX 3 reference-image preparation (pure PIL/numpy; exact parity with the reference is checked by the E1 gate)."""

import mlx.core as mx
import numpy as np
import PIL.Image
import pytest

from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages as Refs


def _image(width: int, height: int) -> PIL.Image.Image:
    rng = np.random.default_rng(0)
    return PIL.Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.fast
class TestFlux3ReferenceImages:
    def test_single_reference_is_padded_to_the_16_grid_and_returns_the_fitted_box(self):
        (ref,), box = Refs.prepare([_image(700, 500)])
        assert ref.size == (704, 512)
        assert box == (2, 6, 702, 506)

    def test_single_reference_box_is_dropped_when_the_user_gave_a_size(self):
        _, box = Refs.prepare([_image(700, 500)], crop_to_fit=False)
        assert box is None

    def test_tiny_reference_pads_symmetrically_to_the_minimum_side(self):
        (ref,), box = Refs.prepare([_image(50, 40)])
        assert ref.size == (64, 64)
        assert box == (7, 12, 57, 52)

    def test_multi_reference_center_crops_and_has_no_box(self):
        refs, box = Refs.prepare([_image(700, 500), _image(300, 1200)])
        assert [r.size for r in refs] == [(688, 496), (288, 1200)]
        assert box is None
        assert Refs.default_size(refs) == (688, 496)

    def test_multi_reference_is_capped_at_one_megapixel(self):
        refs, _ = Refs.prepare([_image(3000, 2000), _image(64, 64)])
        assert refs[0].size == (1248, 832)

    def test_reference_count_is_validated(self):
        with pytest.raises(ValueError):
            Refs.prepare([])
        with pytest.raises(ValueError):
            Refs.prepare([_image(64, 64)] * 11)

    def test_to_array_is_nchw_in_unit_range(self):
        arr = Refs.to_array(_image(32, 16))
        assert arr.shape == (1, 3, 16, 32)
        assert arr.min() >= -1.0 and arr.max() <= 1.0

    def test_reference_ids_carry_the_negative_layer_index(self):
        ids = np.array(Flux3LatentCreator.prepare_ids(2, 3, layer=-2))
        assert ids.shape == (6, 4)
        assert (ids[:, 3] == -2).all()
        assert ids[:, 1].max() == 0.5 and ids[:, 2].max() == 1.0
        assert isinstance(mx.array(ids), mx.array)


def _canvas(size, width=None, height=None):
    references, box = Refs.prepare([_image(*size)])
    return Refs.resolve_canvas(references, box, width, height)


@pytest.mark.fast
class TestFlux3ResolveCanvas:
    def test_no_size_generates_on_the_canvas_and_crops_back(self):
        assert _canvas((700, 500)) == (704, 512, (2, 6, 702, 506))

    def test_the_fitted_size_replayed_from_metadata_crops_back_too(self):
        assert _canvas((700, 500), 700, 500) == (704, 512, (2, 6, 702, 506))

    @pytest.mark.parametrize(("width", "height"), [(512, 512), (700, None), (None, 500), (704, 512)])
    def test_any_other_explicit_size_is_used_as_given_without_a_crop(self, width, height):
        assert _canvas((700, 500), width, height) == (width or 704, height or 512, None)

    def test_an_unpadded_reference_has_nothing_to_crop(self):
        assert _canvas((704, 512), 704, 512) == (704, 512, None)

    def test_multiple_references_never_crop(self):
        references, box = Refs.prepare([_image(700, 500), _image(300, 1200)])
        assert Refs.resolve_canvas(references, box, 688, 496) == (688, 496, None)
