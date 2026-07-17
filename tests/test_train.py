"""
Tests unitaires pour senegal_rental_price.models.train.

On ne teste pas `main()` directement (décoré par @hydra.main, nécessite un
vrai contexte Hydra) mais les fonctions pures qu'il utilise, qui contiennent
toute la logique métier testable.
"""

import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from senegal_rental_price.models.train import evaluate, get_model


class TestGetModel:
    def test_returns_ridge_with_correct_params(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "ridge", "params": {"alpha": 2.0}}})
        model = get_model(cfg)
        assert isinstance(model, Ridge)
        assert model.alpha == 2.0

    def test_returns_random_forest_with_correct_params(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"name": "random_forest", "params": {"n_estimators": 50, "max_depth": 5}}}
        )
        model = get_model(cfg)
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_returns_xgboost_with_correct_params(self) -> None:
        pytest.importorskip("xgboost")
        cfg = OmegaConf.create(
            {"model": {"name": "xgboost", "params": {"n_estimators": 100, "max_depth": 3}}}
        )
        model = get_model(cfg)
        assert model.n_estimators == 100

    def test_raises_for_unknown_model(self) -> None:
        cfg = OmegaConf.create({"model": {"name": "modele_inconnu", "params": {}}})
        with pytest.raises(ValueError):
            get_model(cfg)


class TestEvaluate:
    def test_returns_expected_metrics(self) -> None:
        y_true = pd.Series([100_000, 200_000, 300_000])
        y_pred = [110_000, 190_000, 310_000]

        metrics = evaluate(y_true, y_pred)

        assert set(metrics.keys()) == {"mae", "rmse", "r2"}
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0

    def test_perfect_prediction_gives_zero_error(self) -> None:
        y_true = pd.Series([100_000, 200_000, 300_000])
        y_pred = [100_000, 200_000, 300_000]

        metrics = evaluate(y_true, y_pred)

        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0