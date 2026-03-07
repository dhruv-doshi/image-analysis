import numpy as np
import pytest
import torch
from src.config.metrics import METRICS
from src.models import TechnicalScores, CompositionScores


def test_all_metrics_enabled_by_default():
    # use_heavy_saliency is intentionally disabled by default (fast spectral residual is used instead)
    _OPT_OUT_BY_DEFAULT = {"use_heavy_saliency"}
    assert all(cfg["enabled"] for k, cfg in METRICS.items() if k not in _OPT_OUT_BY_DEFAULT)


def test_disable_brisque_skips_computation(monkeypatch):
    import src.analysis.technical as tech_mod
    monkeypatch.setattr(tech_mod, "is_enabled", lambda m: m != "brisque")
    bgr = np.full((10, 10, 3), 128, dtype=np.uint8)
    tensor = torch.zeros((1, 3, 10, 10), dtype=torch.float32)
    result = tech_mod.analyse(bgr, tensor)
    assert result.brisque is None


def test_weight_used_in_quality_tier(monkeypatch):
    import src.llm.synthesizer as syn
    import src.config.metrics as metrics_mod
    orig = metrics_mod.get_weight
    # Zero out brisque weight so terrible brisque (100.0) does not affect overall
    monkeypatch.setattr(syn, "get_weight", lambda m: 0.0 if m == "brisque" else orig(m))
    tech = TechnicalScores(brisque=100.0)  # all other fields default to None
    comp = CompositionScores()             # all fields default
    result = syn._compute_quality_tier(tech, comp)
    assert result["overall"] != "terrible"
