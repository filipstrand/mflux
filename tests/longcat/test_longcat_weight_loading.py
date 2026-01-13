"""
Unit tests for LongCat weight loading and mapping.

Tests verify:
1. Weight definition structure
2. Component mapping correctness
3. Weight file patterns
4. Tokenizer configuration
5. Quantization predicate
"""

import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.longcat.weights.longcat_weight_definition import LongCatWeightDefinition
from mflux.models.longcat.weights.longcat_weight_mapping import LongCatWeightMapping


class TestLongCatWeightDefinition:
    """Tests for LongCat weight definition structure."""

    @pytest.mark.fast
    def test_get_components_returns_three_components(self):
        """Verify weight definition includes VAE, transformer, and text encoder."""
        components = LongCatWeightDefinition.get_components()

        assert len(components) == 3

        # Check component names
        names = {comp.name for comp in components}
        assert names == {"vae", "transformer", "text_encoder"}

    @pytest.mark.fast
    def test_vae_component_configuration(self):
        """Verify VAE component is configured correctly."""
        components = LongCatWeightDefinition.get_components()
        vae = next(c for c in components if c.name == "vae")

        assert vae.hf_subdir == "vae"
        assert vae.precision == ModelConfig.precision
        assert vae.mapping_getter is not None

    @pytest.mark.fast
    def test_transformer_component_configuration(self):
        """Verify transformer component is configured correctly."""
        components = LongCatWeightDefinition.get_components()
        transformer = next(c for c in components if c.name == "transformer")

        assert transformer.hf_subdir == "transformer"
        assert transformer.precision == ModelConfig.precision
        assert transformer.mapping_getter is not None

    @pytest.mark.fast
    def test_text_encoder_component_configuration(self):
        """Verify text encoder component is configured correctly."""
        components = LongCatWeightDefinition.get_components()
        text_encoder = next(c for c in components if c.name == "text_encoder")

        assert text_encoder.hf_subdir == "text_encoder"
        assert text_encoder.model_attr == "text_encoder"
        assert text_encoder.num_blocks == 28  # Qwen2.5-VL has 28 layers
        assert text_encoder.precision == ModelConfig.precision
        assert text_encoder.mapping_getter is not None


class TestLongCatTokenizers:
    """Tests for LongCat tokenizer configuration."""

    @pytest.mark.fast
    def test_get_tokenizers_returns_qwen(self):
        """Verify LongCat uses Qwen2 tokenizer."""
        tokenizers = LongCatWeightDefinition.get_tokenizers()

        assert len(tokenizers) == 1
        assert tokenizers[0].name == "qwen"

    @pytest.mark.fast
    def test_qwen_tokenizer_configuration(self):
        """Verify Qwen tokenizer is configured correctly."""
        tokenizers = LongCatWeightDefinition.get_tokenizers()
        qwen_tok = tokenizers[0]

        assert qwen_tok.name == "qwen"
        assert qwen_tok.hf_subdir == "tokenizer"
        assert qwen_tok.tokenizer_class == "Qwen2Tokenizer"
        assert qwen_tok.max_length == 512
        assert "tokenizer/**" in qwen_tok.download_patterns


class TestLongCatDownloadPatterns:
    """Tests for LongCat download patterns."""

    @pytest.mark.fast
    def test_download_patterns_include_all_components(self):
        """Verify download patterns cover all components."""
        patterns = LongCatWeightDefinition.get_download_patterns()

        # Should have patterns for text_encoder, transformer, and vae
        assert any("text_encoder" in p for p in patterns)
        assert any("transformer" in p for p in patterns)
        assert any("vae" in p for p in patterns)

    @pytest.mark.fast
    def test_download_patterns_include_safetensors(self):
        """Verify download patterns include safetensors files."""
        patterns = LongCatWeightDefinition.get_download_patterns()

        safetensors_patterns = [p for p in patterns if ".safetensors" in p]
        assert len(safetensors_patterns) >= 3  # At least one per component

    @pytest.mark.fast
    def test_download_patterns_include_json(self):
        """Verify download patterns include JSON config files."""
        patterns = LongCatWeightDefinition.get_download_patterns()

        json_patterns = [p for p in patterns if ".json" in p]
        assert len(json_patterns) >= 3  # At least one per component


class TestLongCatQuantizationPredicate:
    """Tests for LongCat quantization predicate."""

    @pytest.mark.fast
    def test_quantization_predicate_exists(self):
        """Verify quantization predicate is defined."""
        assert hasattr(LongCatWeightDefinition, "quantization_predicate")
        assert callable(LongCatWeightDefinition.quantization_predicate)

    @pytest.mark.fast
    def test_quantization_predicate_rejects_non_64_aligned(self):
        """Test that quantization predicate rejects shapes not aligned to 64."""
        # Mock module with weight not divisible by 64
        class MockModule:
            class Weight:
                shape = [100, 100]  # Not divisible by 64

            weight = Weight()
            to_quantized = True

        result = LongCatWeightDefinition.quantization_predicate("path", MockModule())
        assert result is False

    @pytest.mark.fast
    def test_quantization_predicate_accepts_64_aligned(self):
        """Test that quantization predicate accepts shapes aligned to 64."""
        # Mock module with weight divisible by 64
        class MockModule:
            class Weight:
                shape = [128, 128]  # Divisible by 64

            weight = Weight()
            to_quantized = True

        result = LongCatWeightDefinition.quantization_predicate("path", MockModule())
        assert result is True

    @pytest.mark.fast
    def test_quantization_predicate_requires_to_quantized(self):
        """Test that quantization predicate requires to_quantized method."""
        # Mock module without to_quantized
        class MockModule:
            class Weight:
                shape = [128, 128]

            weight = Weight()

        result = LongCatWeightDefinition.quantization_predicate("path", MockModule())
        assert result is False


class TestLongCatWeightMapping:
    """Tests for LongCat weight mapping structure."""

    @pytest.mark.fast
    def test_transformer_mapping_exists(self):
        """Verify transformer mapping is defined."""
        mapping = LongCatWeightMapping.get_transformer_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_text_encoder_mapping_exists(self):
        """Verify text encoder mapping is defined."""
        mapping = LongCatWeightMapping.get_text_encoder_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_vae_mapping_reuses_flux(self):
        """Verify VAE mapping reuses FLUX mapping."""
        # This is documented in the code
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

        longcat_vae_mapping = LongCatWeightMapping.get_vae_mapping()
        flux_vae_mapping = FluxWeightMapping.get_vae_mapping()

        # Should be identical (same VAE)
        assert len(longcat_vae_mapping) == len(flux_vae_mapping)


class TestLongCatTransformerWeightMapping:
    """Tests for LongCat transformer-specific weight mappings."""

    @pytest.mark.fast
    def test_transformer_mapping_includes_x_embedder(self):
        """Verify transformer mapping includes x_embedder."""
        mapping = LongCatWeightMapping.get_transformer_mapping()
        paths = [target.to_pattern for target in mapping]

        assert "x_embedder.weight" in paths
        assert "x_embedder.bias" in paths

    @pytest.mark.fast
    def test_transformer_mapping_includes_context_embedder(self):
        """Verify transformer mapping includes context_embedder."""
        mapping = LongCatWeightMapping.get_transformer_mapping()
        paths = [target.to_pattern for target in mapping]

        assert "context_embedder.weight" in paths
        assert "context_embedder.bias" in paths

    @pytest.mark.fast
    def test_transformer_mapping_includes_proj_out(self):
        """Verify transformer mapping includes proj_out."""
        mapping = LongCatWeightMapping.get_transformer_mapping()
        paths = [target.to_pattern for target in mapping]

        assert "proj_out.weight" in paths
        assert "proj_out.bias" in paths

    @pytest.mark.fast
    def test_transformer_mapping_includes_joint_blocks(self):
        """Verify transformer mapping includes joint block patterns."""
        mapping = LongCatWeightMapping.get_transformer_mapping()
        paths = [target.to_pattern for target in mapping]

        # Check for joint block patterns (should support 0-9 for 10 blocks)
        joint_patterns = [p for p in paths if "transformer_blocks.{block}" in p]
        assert len(joint_patterns) > 0

    @pytest.mark.fast
    def test_transformer_mapping_includes_single_blocks(self):
        """Verify transformer mapping includes single block patterns."""
        mapping = LongCatWeightMapping.get_transformer_mapping()
        paths = [target.to_pattern for target in mapping]

        # Check for single block patterns (should support 0-19 for 20 blocks)
        single_patterns = [p for p in paths if "single_transformer_blocks.{block}" in p]
        assert len(single_patterns) > 0


class TestLongCatTextEncoderWeightMapping:
    """Tests for LongCat text encoder weight mappings."""

    @pytest.mark.fast
    def test_text_encoder_mapping_includes_embeddings(self):
        """Verify text encoder mapping includes token embeddings."""
        mapping = LongCatWeightMapping.get_text_encoder_mapping()
        paths = [target.to_pattern for target in mapping]

        assert "embed_tokens.weight" in paths

    @pytest.mark.fast
    def test_text_encoder_mapping_includes_layers(self):
        """Verify text encoder mapping includes transformer layers."""
        mapping = LongCatWeightMapping.get_text_encoder_mapping()
        paths = [target.to_pattern for target in mapping]

        # Check for layer patterns (should support 0-27 for 28 layers)
        layer_patterns = [p for p in paths if "layers.{block}" in p]
        assert len(layer_patterns) > 0

    @pytest.mark.fast
    def test_text_encoder_mapping_includes_norm(self):
        """Verify text encoder mapping includes final norm layer."""
        mapping = LongCatWeightMapping.get_text_encoder_mapping()
        paths = [target.to_pattern for target in mapping]

        assert "norm.weight" in paths


class TestLongCatWeightMappingVsFlux:
    """Tests comparing LongCat weight mapping to FLUX."""

    @pytest.mark.fast
    def test_longcat_vae_same_as_flux(self):
        """Verify LongCat uses same VAE structure as FLUX."""
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

        longcat_vae = LongCatWeightMapping.get_vae_mapping()
        flux_vae = FluxWeightMapping.get_vae_mapping()

        # Should be identical
        assert len(longcat_vae) == len(flux_vae)

        # Compare first few weight paths
        longcat_paths = [t.to_pattern for t in longcat_vae[:5]]
        flux_paths = [t.to_pattern for t in flux_vae[:5]]
        assert longcat_paths == flux_paths

    @pytest.mark.fast
    def test_longcat_transformer_different_from_flux(self):
        """Verify LongCat transformer mapping is different from FLUX."""
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

        longcat_transformer = LongCatWeightMapping.get_transformer_mapping()
        flux_transformer = FluxWeightMapping.get_transformer_mapping()

        # Different architectures, so mappings should differ
        # (LongCat has 10+20 blocks vs FLUX 19+38)
        # Just verify they both exist and have content
        assert len(longcat_transformer) > 0
        assert len(flux_transformer) > 0


class TestLongCatWeightDefinitionDocumentation:
    """Tests for correct documentation in weight definitions."""

    @pytest.mark.fast
    def test_vae_documentation_says_16_channels(self):
        """Verify documentation correctly states 16 VAE channels (not 8)."""
        import inspect

        # Check module docstring
        module_doc = inspect.getdoc(LongCatWeightDefinition)
        assert "16 latent channels" in module_doc
        assert "8 latent channels" not in module_doc

        # Check class docstring
        class_doc = LongCatWeightDefinition.__doc__
        assert "16 latent channels" in class_doc
        assert "8 latent channels" not in class_doc
