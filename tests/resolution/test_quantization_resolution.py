import pytest

from mflux.models.common.resolution.quantization_resolution import QuantizationResolution


class TestDecideQuantization:
    @pytest.mark.fast
    def test_no_quantization_when_neither_specified(self):
        bits, warning = QuantizationResolution.resolve(stored=None, requested=None)
        assert bits is None
        assert warning is None

    @pytest.mark.fast
    @pytest.mark.parametrize("requested_bits", [3, 4, 5, 6, 8])
    def test_on_the_fly_quantization(self, requested_bits):
        bits, warning = QuantizationResolution.resolve(stored=None, requested=requested_bits)
        assert bits == requested_bits
        assert warning is None

    @pytest.mark.fast
    @pytest.mark.parametrize("stored_bits", [3, 4, 5, 6, 8])
    def test_prequantized_no_request(self, stored_bits):
        bits, warning = QuantizationResolution.resolve(stored=stored_bits, requested=None)
        assert bits == stored_bits
        assert warning is None

    @pytest.mark.fast
    @pytest.mark.parametrize("bits_value", [3, 4, 5, 6, 8])
    def test_prequantized_matching_request(self, bits_value):
        bits, warning = QuantizationResolution.resolve(stored=bits_value, requested=bits_value)
        assert bits == bits_value
        assert warning is None

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "stored_bits,requested_bits",
        [(4, 8), (8, 4), (4, 3), (3, 8), (6, 4)],
    )
    def test_prequantized_conflicting_request_uses_stored(self, stored_bits, requested_bits):
        bits, warning = QuantizationResolution.resolve(stored=stored_bits, requested=requested_bits)
        assert bits == stored_bits

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "stored_bits,requested_bits",
        [(4, 8), (8, 4), (4, 3)],
    )
    def test_prequantized_conflicting_request_warns(self, stored_bits, requested_bits):
        bits, warning = QuantizationResolution.resolve(stored=stored_bits, requested=requested_bits)
        assert warning is not None
        assert f"{stored_bits}-bit" in warning
        assert f"-q {requested_bits}" in warning


class TestQuantizationPolicyCompleteness:
    VALID_QUANT_LEVELS = [None, 3, 4, 5, 6, 8]

    @pytest.mark.fast
    def test_all_combinations_handled(self):
        for stored in self.VALID_QUANT_LEVELS:
            for requested in self.VALID_QUANT_LEVELS:
                bits, warning = QuantizationResolution.resolve(stored=stored, requested=requested)
                assert bits is None or bits in self.VALID_QUANT_LEVELS

    @pytest.mark.fast
    def test_result_is_always_stored_or_requested_or_none(self):
        for stored in self.VALID_QUANT_LEVELS:
            for requested in self.VALID_QUANT_LEVELS:
                bits, _ = QuantizationResolution.resolve(stored=stored, requested=requested)
                assert bits in (stored, requested, None)
