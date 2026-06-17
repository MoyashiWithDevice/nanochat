"""
Tests for nanochat.common - common utilities.

Run: python -m pytest tests/test_common.py -v
"""

import os
import logging
import pytest
import torch
from unittest.mock import patch

from nanochat.common import (
    get_base_dir,
    print0,
    is_ddp_requested,
    is_ddp_initialized,
    get_dist_info,
    autodetect_device_type,
    get_peak_flops,
    ColoredFormatter,
    DummyWandb,
    _detect_compute_dtype,
)


class TestGetBaseDir:
    """Test get_base_dir utility."""

    def test_returns_existing_directory(self):
        base_dir = get_base_dir()
        assert os.path.isdir(base_dir)

    def test_respects_env_var(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            custom_dir = os.path.join(td, "custom_nanochat")
            with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": custom_dir}):
                result = get_base_dir()
                assert result == custom_dir
                assert os.path.isdir(result)

    def test_default_is_in_cache(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove NANOCHAT_BASE_DIR if set
            env = os.environ.copy()
            env.pop("NANOCHAT_BASE_DIR", None)
            with patch.dict(os.environ, env, clear=True):
                result = get_base_dir()
                assert ".cache" in result or "NANOCHAT_BASE_DIR" in os.environ


class TestPrint0:
    """Test print0 - only prints on rank 0."""

    def test_prints_on_rank_0(self, capsys):
        with patch.dict(os.environ, {"RANK": "0"}):
            print0("hello")
            captured = capsys.readouterr()
            assert "hello" in captured.out

    def test_silent_on_rank_1(self, capsys):
        with patch.dict(os.environ, {"RANK": "1"}):
            print0("hello")
            captured = capsys.readouterr()
            assert captured.out == ""

    def test_prints_when_no_rank(self, capsys):
        env = os.environ.copy()
        env.pop("RANK", None)
        with patch.dict(os.environ, env, clear=True):
            print0("hello")
            captured = capsys.readouterr()
            assert "hello" in captured.out


class TestDDPUtils:
    """Test DDP-related utility functions."""

    def test_is_ddp_requested_false(self):
        env = os.environ.copy()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            env.pop(k, None)
        with patch.dict(os.environ, env, clear=True):
            assert is_ddp_requested() is False

    def test_is_ddp_requested_true(self):
        with patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}):
            assert is_ddp_requested() is True

    def test_is_ddp_initialized_false(self):
        # In a non-DDP test environment, this should be False
        assert is_ddp_initialized() is False

    def test_get_dist_info_single_gpu(self):
        env = os.environ.copy()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            env.pop(k, None)
        with patch.dict(os.environ, env, clear=True):
            ddp, rank, local_rank, world_size = get_dist_info()
            assert ddp is False
            assert rank == 0
            assert local_rank == 0
            assert world_size == 1

    def test_get_dist_info_ddp_env(self):
        with patch.dict(os.environ, {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}):
            ddp, rank, local_rank, world_size = get_dist_info()
            assert ddp is True
            assert rank == 3
            assert local_rank == 1
            assert world_size == 8


class TestAutodetectDeviceType:
    """Test autodetect_device_type function."""

    def test_returns_valid_device(self):
        device_type = autodetect_device_type()
        assert device_type in ("cuda", "mps", "cpu")

    def test_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                device_type = autodetect_device_type()
                assert device_type == "cpu"


class TestGetPeakFlops:
    """Test get_peak_flops for various GPU names."""

    def test_h100_sxm(self):
        flops = get_peak_flops("NVIDIA H100 80GB HBM3")
        assert flops == 989e12

    def test_h100_pcie(self):
        flops = get_peak_flops("NVIDIA H100 PCIe")
        assert flops == 756e12

    def test_a100(self):
        flops = get_peak_flops("NVIDIA A100-SXM4-80GB")
        assert flops == 312e12

    def test_4090(self):
        flops = get_peak_flops("NVIDIA GeForce RTX 4090")
        assert flops == 165.2e12

    def test_unknown_gpu_returns_inf(self):
        flops = get_peak_flops("Unknown GPU XYZ")
        assert flops == float('inf')

    def test_case_insensitive(self):
        flops1 = get_peak_flops("nvidia h100 80gb hbm3")
        flops2 = get_peak_flops("NVIDIA H100 80GB HBM3")
        assert flops1 == flops2


class TestDetectComputeDtype:
    """Test _detect_compute_dtype."""

    def test_env_override_bfloat16(self):
        with patch.dict(os.environ, {"NANOCHAT_DTYPE": "bfloat16"}):
            dtype, reason = _detect_compute_dtype()
            assert dtype == torch.bfloat16
            assert "NANOCHAT_DTYPE" in reason

    def test_env_override_float32(self):
        with patch.dict(os.environ, {"NANOCHAT_DTYPE": "float32"}):
            dtype, reason = _detect_compute_dtype()
            assert dtype == torch.float32

    def test_no_cuda_returns_float32(self):
        env = os.environ.copy()
        env.pop("NANOCHAT_DTYPE", None)
        with patch.dict(os.environ, env, clear=True):
            with patch("torch.cuda.is_available", return_value=False):
                dtype, reason = _detect_compute_dtype()
                assert dtype == torch.float32
                assert "no CUDA" in reason


class TestColoredFormatter:
    """Test ColoredFormatter."""

    def test_format_info(self):
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Processing 100 docs", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert "100 docs" in formatted

    def test_format_error(self):
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="Something failed", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert "Something failed" in formatted


class TestDummyWandb:
    """Test DummyWandb class."""

    def test_log_does_nothing(self):
        wandb = DummyWandb()
        wandb.log({"loss": 0.5}, step=1)  # Should not raise

    def test_finish_does_nothing(self):
        wandb = DummyWandb()
        wandb.finish()  # Should not raise
