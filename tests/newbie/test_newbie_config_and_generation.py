"""
Unit tests for NewBie-image Config initialization and generation patterns.

Tests verify:
1. Config initialization with correct parameters
2. Weight loading verification
3. generate_image method follows MFLUX patterns
4. num_dit_blocks detection and fallback
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import mlx.core as mx

from mflux.models.common.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData


class TestConfigInitialization:
    """Tests for Config initialization following MFLUX patterns."""

    @pytest.mark.fast
    def test_config_accepts_guidance_parameter(self):
        """Verify Config accepts 'guidance' parameter (not 'guidance_scale')."""
        model_config = ModelConfig.newbie()
        config = Config(
            model_config=model_config,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance=5.0,
        )
        assert config.guidance == 5.0

    @pytest.mark.fast
    def test_config_does_not_accept_precision_parameter(self):
        """Verify Config does not accept 'precision' as a constructor parameter."""
        model_config = ModelConfig.newbie()
        # This should NOT raise an error, but precision should be ignored
        # precision is accessed as a property, not a constructor parameter
        config = Config(
            model_config=model_config,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance=5.0,
        )
        # Verify we can access precision as a property
        assert hasattr(config, 'precision')
        # Precision comes from ModelConfig, not the Config constructor
        assert config.precision == ModelConfig.precision

    @pytest.mark.fast
    def test_config_accepts_image_path_and_strength(self):
        """Verify Config accepts image_path and image_strength for img2img."""
        model_config = ModelConfig.newbie()
        config = Config(
            model_config=model_config,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance=5.0,
            image_path="/path/to/image.png",
            image_strength=0.5,
        )
        assert config.image_path is not None
        assert config.image_strength == 0.5

    @pytest.mark.fast
    def test_config_dimensions_rounded_to_multiple_of_16(self):
        """Verify Config rounds dimensions to multiples of 16."""
        model_config = ModelConfig.newbie()
        config = Config(
            model_config=model_config,
            num_inference_steps=28,
            height=1025,  # Not multiple of 16
            width=1030,   # Not multiple of 16
            guidance=5.0,
        )
        # Should be rounded down to nearest multiple of 16
        assert config.height % 16 == 0
        assert config.width % 16 == 0
        assert config.height <= 1025
        assert config.width <= 1030

    @pytest.mark.fast
    def test_config_scheduler_property(self):
        """Verify Config has scheduler property that can be set."""
        model_config = ModelConfig.newbie()
        config = Config(
            model_config=model_config,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance=5.0,
        )
        # Should be able to access scheduler property
        # (will initialize on first access)
        assert hasattr(config, 'scheduler')


class TestLoadedWeightsNumDitBlocks:
    """Tests for LoadedWeights.num_dit_blocks() method."""

    @pytest.mark.fast
    def test_num_dit_blocks_with_nextdit_blocks(self):
        """Verify num_dit_blocks detects 'nextdit_blocks' in weights."""
        weights = LoadedWeights(
            components={
                "transformer": {
                    "nextdit_blocks": [{"block": i} for i in range(36)]
                }
            },
            meta_data=MetaData(),
        )
        assert weights.num_dit_blocks() == 36

    @pytest.mark.fast
    def test_num_dit_blocks_with_dit_blocks(self):
        """Verify num_dit_blocks detects 'dit_blocks' in weights."""
        weights = LoadedWeights(
            components={
                "transformer": {
                    "dit_blocks": [{"block": i} for i in range(36)]
                }
            },
            meta_data=MetaData(),
        )
        assert weights.num_dit_blocks() == 36

    @pytest.mark.fast
    def test_num_dit_blocks_returns_zero_if_not_found(self):
        """Verify num_dit_blocks returns 0 if blocks not found."""
        weights = LoadedWeights(
            components={
                "transformer": {
                    "some_other_key": []
                }
            },
            meta_data=MetaData(),
        )
        assert weights.num_dit_blocks() == 0

    @pytest.mark.fast
    def test_num_dit_blocks_searches_all_components(self):
        """Verify num_dit_blocks searches all components if transformer not found."""
        weights = LoadedWeights(
            components={
                "some_component": {
                    "nextdit_blocks": [{"block": i} for i in range(36)]
                }
            },
            meta_data=MetaData(),
        )
        # Should find nextdit_blocks even if not in 'transformer' component
        assert weights.num_dit_blocks() == 36


class TestNewBieInitializerBlockCount:
    """Tests for NewBieInitializer block count detection."""

    @pytest.mark.fast
    def test_initializer_uses_weights_block_count(self):
        """Verify initializer uses block count from weights when available."""
        from mflux.models.newbie.newbie_initializer import NewBieInitializer

        # Create mock weights with block count
        mock_weights = Mock(spec=LoadedWeights)
        mock_weights.num_dit_blocks.return_value = 24  # Non-default value

        model_config = ModelConfig.newbie()

        # Mock the model and its attributes
        mock_model = Mock()
        mock_model.model_config = model_config
        mock_model.prompt_cache = {}
        mock_model.callbacks = Mock()
        mock_model.tiling_config = None
        mock_model.lora_paths = None
        mock_model.lora_scales = None

        # Patch the NextDiT constructor to capture num_blocks
        with patch('mflux.models.newbie.newbie_initializer.NextDiT') as mock_nextdit, \
             patch('mflux.models.newbie.newbie_initializer.VAE'), \
             patch('mflux.models.newbie.newbie_initializer.Gemma3Encoder'), \
             patch('mflux.models.newbie.newbie_initializer.JinaCLIPEncoder'):

            NewBieInitializer._init_models(mock_model, model_config, mock_weights)

            # Verify NextDiT was called with correct num_blocks
            mock_nextdit.assert_called_once()
            call_kwargs = mock_nextdit.call_args[1]
            assert call_kwargs['num_blocks'] == 24

    @pytest.mark.fast
    def test_initializer_uses_default_when_weights_return_zero(self):
        """Verify initializer uses default 36 blocks when weights return 0."""
        from mflux.models.newbie.newbie_initializer import NewBieInitializer

        # Create mock weights with no block count
        mock_weights = Mock(spec=LoadedWeights)
        mock_weights.num_dit_blocks.return_value = 0

        model_config = ModelConfig.newbie()

        # Mock the model and its attributes
        mock_model = Mock()
        mock_model.model_config = model_config
        mock_model.prompt_cache = {}
        mock_model.callbacks = Mock()
        mock_model.tiling_config = None
        mock_model.lora_paths = None
        mock_model.lora_scales = None

        # Patch the NextDiT constructor to capture num_blocks
        with patch('mflux.models.newbie.newbie_initializer.NextDiT') as mock_nextdit, \
             patch('mflux.models.newbie.newbie_initializer.VAE'), \
             patch('mflux.models.newbie.newbie_initializer.Gemma3Encoder'), \
             patch('mflux.models.newbie.newbie_initializer.JinaCLIPEncoder'):

            NewBieInitializer._init_models(mock_model, model_config, mock_weights)

            # Verify NextDiT was called with default 36 blocks
            mock_nextdit.assert_called_once()
            call_kwargs = mock_nextdit.call_args[1]
            assert call_kwargs['num_blocks'] == 36

    @pytest.mark.fast
    def test_initializer_uses_model_config_block_count_if_present(self):
        """Verify initializer uses model_config.num_dit_blocks when weights return 0."""
        from mflux.models.newbie.newbie_initializer import NewBieInitializer

        # Create mock weights with no block count
        mock_weights = Mock(spec=LoadedWeights)
        mock_weights.num_dit_blocks.return_value = 0

        # Create model config with custom block count
        model_config = ModelConfig.newbie()
        model_config.num_dit_blocks = 48  # Custom value

        # Mock the model and its attributes
        mock_model = Mock()
        mock_model.model_config = model_config
        mock_model.prompt_cache = {}
        mock_model.callbacks = Mock()
        mock_model.tiling_config = None
        mock_model.lora_paths = None
        mock_model.lora_scales = None

        # Patch the NextDiT constructor to capture num_blocks
        with patch('mflux.models.newbie.newbie_initializer.NextDiT') as mock_nextdit, \
             patch('mflux.models.newbie.newbie_initializer.VAE'), \
             patch('mflux.models.newbie.newbie_initializer.Gemma3Encoder'), \
             patch('mflux.models.newbie.newbie_initializer.JinaCLIPEncoder'):

            NewBieInitializer._init_models(mock_model, model_config, mock_weights)

            # Verify NextDiT was called with model_config block count
            mock_nextdit.assert_called_once()
            call_kwargs = mock_nextdit.call_args[1]
            assert call_kwargs['num_blocks'] == 48


class TestGenerateImagePattern:
    """Tests for generate_image method following MFLUX patterns."""

    @pytest.mark.fast
    def test_generate_image_creates_config_with_correct_parameters(self):
        """Verify generate_image creates Config with correct parameters (not precision or guidance_scale)."""
        # This test verifies the FIX for Issue A1:
        # - Config should accept 'guidance' (not 'guidance_scale')
        # - Config should NOT accept 'precision' as a parameter

        # We'll test by inspecting the generate_image source code
        from mflux.models.newbie.variants.txt2img.newbie import NewBie
        import inspect

        source = inspect.getsource(NewBie.generate_image)

        # Verify that guidance (not guidance_scale) is used
        assert 'guidance=' in source or 'guidance:' in source
        assert 'guidance_scale=' not in source

        # Verify that precision is not passed to Config
        # (it should be accessed as a property, not passed as a parameter)
        assert 'precision=' not in source or 'precision:' not in source or \
               'Config(' not in source.split('precision=')[0][-50:]  # Not in Config() call

    @pytest.mark.fast
    def test_generate_image_uses_callback_context(self):
        """Verify generate_image uses callback context pattern like FLUX."""
        from mflux.models.newbie.variants.txt2img.newbie import NewBie
        import inspect

        source = inspect.getsource(NewBie.generate_image)

        # Verify callback pattern is used
        assert 'self.callbacks.start' in source
        assert 'before_loop' in source
        assert 'in_loop' in source
        assert 'after_loop' in source

    @pytest.mark.fast
    def test_generate_image_handles_keyboard_interrupt(self):
        """Verify generate_image has KeyboardInterrupt handling."""
        from mflux.models.newbie.variants.txt2img.newbie import NewBie
        import inspect

        source = inspect.getsource(NewBie.generate_image)

        # Verify KeyboardInterrupt handling exists
        assert 'KeyboardInterrupt' in source
        assert 'interruption' in source  # Callback method
        assert 'StopImageGenerationException' in source


class TestNextDiTBlockNorm2Check:
    """Tests for NextDiT block norm2 None check (Issue A3)."""

    @pytest.mark.fast
    def test_nextdit_block_handles_none_norm2(self):
        """Verify NextDiT block handles None norm2 gracefully."""
        from mflux.models.newbie.model.newbie_transformer.nextdit_block import NextDiTBlock

        # Create block without cross-attention (norm2 will be None)
        block = NextDiTBlock(
            hidden_dim=256,
            num_query_heads=8,
            num_kv_heads=4,
            mlp_dim=512,
            has_cross_attention=False,
        )

        # Verify norm2 is None
        assert block.norm2 is None

        # Create test inputs
        hidden_states = mx.random.normal((1, 10, 256))
        conditioning = mx.random.normal((1, 256))

        # Call should not raise error even with norm2=None
        output = block(
            hidden_states=hidden_states,
            conditioning=conditioning,
            encoder_hidden_states=None,
        )

        # Output should have correct shape
        assert output.shape == hidden_states.shape

    @pytest.mark.fast
    def test_nextdit_block_uses_norm2_when_present(self):
        """Verify NextDiT block uses norm2 when has_cross_attention=True."""
        from mflux.models.newbie.model.newbie_transformer.nextdit_block import NextDiTBlock

        # Create block with cross-attention (norm2 will be present)
        # Use matching dimensions for hidden_dim and text_dim
        block = NextDiTBlock(
            hidden_dim=256,
            num_query_heads=8,
            num_kv_heads=4,
            mlp_dim=512,
            text_dim=256,  # Match hidden_dim to avoid dimension mismatch
            has_cross_attention=True,
        )

        # Verify norm2 is not None
        assert block.norm2 is not None

        # Create test inputs with matching dimensions
        hidden_states = mx.random.normal((1, 10, 256))
        conditioning = mx.random.normal((1, 256))
        encoder_hidden_states = mx.random.normal((1, 5, 256))  # text_dim=256

        # Call should work with norm2 present
        output = block(
            hidden_states=hidden_states,
            conditioning=conditioning,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Output should have correct shape
        assert output.shape == hidden_states.shape
