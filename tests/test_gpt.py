"""
Tests for nanochat.gpt - GPT model architecture.

Run: python -m pytest tests/test_gpt.py -v
"""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig, norm, apply_rotary_emb, has_ve, Linear, MLP, Block


class TestGPTConfig:
    """Test GPTConfig dataclass."""

    def test_default_config(self):
        config = GPTConfig()
        assert config.sequence_len == 2048
        assert config.vocab_size == 32768
        assert config.n_layer == 12
        assert config.n_head == 6
        assert config.n_kv_head == 6
        assert config.n_embd == 768
        assert config.window_pattern == "SSSL"

    def test_custom_config(self):
        config = GPTConfig(n_layer=4, n_head=4, n_kv_head=2, n_embd=256)
        assert config.n_layer == 4
        assert config.n_head == 4
        assert config.n_kv_head == 2
        assert config.n_embd == 256


class TestNorm:
    """Test RMS norm function."""

    def test_norm_shape_preserved(self):
        x = torch.randn(2, 16, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_norm_unit_rms(self):
        x = torch.randn(4, 32, 128)
        y = norm(x)
        # After RMS norm, the RMS of the last dimension should be ~1
        rms = (y ** 2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestApplyRotaryEmb:
    """Test rotary embedding application."""

    def test_shape_preserved(self):
        B, T, H, D = 2, 16, 4, 32
        x = torch.randn(B, T, H, D)
        cos = torch.randn(1, T, 1, D // 2)
        sin = torch.randn(1, T, 1, D // 2)
        y = apply_rotary_emb(x, cos, sin)
        assert y.shape == x.shape

    def test_zero_rotation_is_identity(self):
        B, T, H, D = 2, 8, 4, 16
        x = torch.randn(B, T, H, D)
        cos = torch.ones(1, T, 1, D // 2)
        sin = torch.zeros(1, T, 1, D // 2)
        y = apply_rotary_emb(x, cos, sin)
        assert torch.allclose(y, x, atol=1e-6)


class TestHasVE:
    """Test has_ve helper function."""

    def test_alternating_pattern(self):
        # For n_layer=4 (last_layer=3, 3%2=1): ve on layers 1, 3
        results = [has_ve(i, 4) for i in range(4)]
        assert results == [False, True, False, True]

    def test_last_layer_always_has_ve(self):
        for n_layer in [1, 2, 3, 4, 6, 12]:
            assert has_ve(n_layer - 1, n_layer) is True


class TestLinear:
    """Test custom Linear layer that casts weights."""

    def test_forward_matches_dtype(self):
        lin = Linear(64, 32, bias=False)
        x = torch.randn(2, 8, 64)
        y = lin(x)
        assert y.shape == (2, 8, 32)
        assert y.dtype == x.dtype


class TestMLP:
    """Test MLP module."""

    def test_forward_shape(self):
        config = GPTConfig(n_embd=64)
        mlp = MLP(config)
        x = torch.randn(2, 16, 64)
        y = mlp(x)
        assert y.shape == x.shape

    def test_output_is_differentiable(self):
        config = GPTConfig(n_embd=64)
        mlp = MLP(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestGPT:
    """Test GPT model."""

    @pytest.fixture
    def small_config(self):
        return GPTConfig(
            sequence_len=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        )

    @pytest.fixture
    def small_model(self, small_config):
        model = GPT(small_config)
        model.init_weights()
        return model

    def test_model_creation(self, small_model, small_config):
        assert small_model.config == small_config

    def test_forward_shape(self, small_model):
        B, T = 2, 16
        x = torch.randint(0, 256, (B, T))
        logits = small_model(x)
        # logits should be (B, T, vocab_size)
        assert logits.shape == (B, T, 256)

    def test_forward_with_targets(self, small_model):
        B, T = 2, 16
        x = torch.randint(0, 256, (B, T))
        y = torch.randint(0, 256, (B, T))
        loss = small_model(x, y)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_num_scaling_params(self, small_model):
        params = small_model.num_scaling_params()
        assert 'total' in params
        assert 'wte' in params
        assert 'transformer_matrices' in params
        assert params['total'] == sum(p.numel() for p in small_model.parameters())

    def test_estimate_flops(self, small_model):
        flops = small_model.estimate_flops()
        assert flops > 0

    def test_get_device(self, small_model):
        device = small_model.get_device()
        assert device == torch.device("cpu")

    def test_window_sizes_all_long(self):
        config = GPTConfig(n_layer=4, sequence_len=128, window_pattern="L")
        model = GPT(config)
        # All layers should have full context
        for ws in model.window_sizes:
            assert ws == (128, 0)

    def test_window_sizes_sssl_pattern(self):
        config = GPTConfig(n_layer=8, sequence_len=2048, window_pattern="SSSL")
        model = GPT(config)
        # Last layer should always be long
        assert model.window_sizes[-1] == (2048, 0)
        # First layer should be short
        short_window = -(-2048 // 4 // 128) * 128  # ceil to tile size
        assert model.window_sizes[0] == (short_window, 0)

    def test_window_pattern_last_layer_always_long(self):
        config = GPTConfig(n_layer=4, sequence_len=512, window_pattern="S")
        model = GPT(config)
        # Despite pattern being all S, last layer is forced to L
        assert model.window_sizes[-1] == (512, 0)

    def test_loss_reduction_none(self, small_model):
        B, T = 2, 8
        x = torch.randint(0, 256, (B, T))
        y = torch.randint(0, 256, (B, T))
        loss = small_model(x, y, loss_reduction='none')
        # cross_entropy with reduction='none' on flattened inputs returns (B*T,)
        assert loss.shape == (B * T,)

    def test_gqa_model(self):
        """Test model with GQA (fewer KV heads than Q heads)."""
        config = GPTConfig(
            sequence_len=32,
            vocab_size=128,
            n_layer=2,
            n_head=8,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        )
        model = GPT(config)
        model.init_weights()
        x = torch.randint(0, 128, (1, 8))
        logits = model(x)
        assert logits.shape == (1, 8, 128)
