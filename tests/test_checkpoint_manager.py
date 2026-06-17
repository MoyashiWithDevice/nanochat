"""
Tests for nanochat.checkpoint_manager - checkpoint save/load utilities.

Run: python -m pytest tests/test_checkpoint_manager.py -v
"""

import os
import json
import tempfile
import pytest
import torch

from nanochat.checkpoint_manager import (
    save_checkpoint,
    load_checkpoint,
    find_largest_model,
    find_last_step,
    _patch_missing_config_keys,
    _patch_missing_keys,
)
from nanochat.gpt import GPTConfig


class TestSaveLoadCheckpoint:
    """Test checkpoint save/load roundtrip."""

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as td:
            # Create dummy model data
            model_data = {
                "weight1": torch.randn(10, 10),
                "weight2": torch.randn(5, 5),
            }
            meta_data = {
                "model_config": {"n_layer": 4, "n_head": 4},
                "step": 100,
            }

            save_checkpoint(td, step=100, model_data=model_data, optimizer_data=None, meta_data=meta_data)

            # Verify files exist
            assert os.path.exists(os.path.join(td, "model_000100.pt"))
            assert os.path.exists(os.path.join(td, "meta_000100.json"))

            # Load and verify
            loaded_model, loaded_opt, loaded_meta = load_checkpoint(td, step=100, device=torch.device("cpu"))
            assert loaded_opt is None
            assert loaded_meta["step"] == 100
            assert torch.allclose(loaded_model["weight1"], model_data["weight1"])
            assert torch.allclose(loaded_model["weight2"], model_data["weight2"])

    def test_save_and_load_with_optimizer(self):
        with tempfile.TemporaryDirectory() as td:
            model_data = {"w": torch.randn(3, 3)}
            optimizer_data = {"state": torch.randn(3, 3)}
            meta_data = {"step": 50}

            save_checkpoint(td, step=50, model_data=model_data, optimizer_data=optimizer_data, meta_data=meta_data)

            # Verify optimizer file exists
            assert os.path.exists(os.path.join(td, "optim_000050_rank0.pt"))

            # Load with optimizer
            loaded_model, loaded_opt, loaded_meta = load_checkpoint(
                td, step=50, device=torch.device("cpu"), load_optimizer=True
            )
            assert loaded_opt is not None
            assert torch.allclose(loaded_opt["state"], optimizer_data["state"])

    def test_non_rank0_does_not_save_model(self):
        with tempfile.TemporaryDirectory() as td:
            model_data = {"w": torch.randn(3, 3)}
            meta_data = {"step": 10}
            optimizer_data = {"state": torch.randn(3, 3)}

            save_checkpoint(td, step=10, model_data=model_data, optimizer_data=optimizer_data, meta_data=meta_data, rank=1)

            # rank=1 should NOT save model/meta, but SHOULD save optimizer with rank suffix
            assert not os.path.exists(os.path.join(td, "model_000010.pt"))
            assert not os.path.exists(os.path.join(td, "meta_000010.json"))
            assert os.path.exists(os.path.join(td, "optim_000010_rank1.pt"))


class TestFindLargestModel:
    """Test find_largest_model."""

    def test_finds_largest_depth(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "d4"))
            os.makedirs(os.path.join(td, "d12"))
            os.makedirs(os.path.join(td, "d26"))
            os.makedirs(os.path.join(td, "d8"))

            result = find_largest_model(td)
            assert result == "d26"

    def test_raises_on_empty(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                find_largest_model(td)

    def test_non_d_pattern_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            # Create directories that don't match d<number> pattern
            os.makedirs(os.path.join(td, "custom_model"))
            result = find_largest_model(td)
            assert result == "custom_model"


class TestFindLastStep:
    """Test find_last_step."""

    def test_finds_largest_step(self):
        with tempfile.TemporaryDirectory() as td:
            # Create dummy checkpoint files
            for step in [100, 200, 500, 1000]:
                torch.save({}, os.path.join(td, f"model_{step:06d}.pt"))

            result = find_last_step(td)
            assert result == 1000

    def test_raises_on_empty(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                find_last_step(td)


class TestPatchMissingConfigKeys:
    """Test _patch_missing_config_keys for backward compatibility."""

    def test_adds_window_pattern_if_missing(self):
        config = {"n_layer": 4, "n_head": 4}
        _patch_missing_config_keys(config)
        assert config["window_pattern"] == "L"

    def test_does_not_overwrite_existing(self):
        config = {"n_layer": 4, "window_pattern": "SSSL"}
        _patch_missing_config_keys(config)
        assert config["window_pattern"] == "SSSL"


class TestPatchMissingKeys:
    """Test _patch_missing_keys for backward compatibility."""

    def test_adds_resid_lambdas_if_missing(self):
        model_data = {}
        config = GPTConfig(n_layer=4)
        _patch_missing_keys(model_data, config)
        assert "resid_lambdas" in model_data
        assert model_data["resid_lambdas"].shape == (4,)
        assert torch.all(model_data["resid_lambdas"] == 1.0)

    def test_adds_x0_lambdas_if_missing(self):
        model_data = {}
        config = GPTConfig(n_layer=6)
        _patch_missing_keys(model_data, config)
        assert "x0_lambdas" in model_data
        assert model_data["x0_lambdas"].shape == (6,)
        assert torch.all(model_data["x0_lambdas"] == 0.0)

    def test_does_not_overwrite_existing(self):
        existing = torch.tensor([0.5, 0.6, 0.7, 0.8])
        model_data = {"resid_lambdas": existing, "x0_lambdas": existing}
        config = GPTConfig(n_layer=4)
        _patch_missing_keys(model_data, config)
        assert torch.allclose(model_data["resid_lambdas"], existing)
