"""
Bayesian Model Averaging (BMA) Ensemble
Weight models by their posterior probability of being correct.
Automatically handles model uncertainty and adapts weights over time.
"""
import numpy as np
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

BMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bayesian_ensemble.pkl")


class BayesianModelAveraging:
    def __init__(self, n_models=5, alpha_prior=1.0):
        self.n_models = n_models
        self.alpha_prior = alpha_prior
        self.weights = np.ones(n_models) / n_models
        self.alphas = np.ones(n_models) * alpha_prior
        self.betas = np.ones(n_models) * alpha_prior
        self.total_predictions = 0
        self.model_names = [f"model_{i}" for i in range(n_models)]
        self.is_fitted = False

    def set_model_names(self, names):
        self.model_names = names
        self.n_models = len(names)
        self.alphas = np.ones(self.n_models) * self.alpha_prior
        self.betas = np.ones(self.n_models) * self.alpha_prior
        self.weights = np.ones(self.n_models) / self.n_models

    def update(self, predictions, actual_direction):
        for i in range(self.n_models):
            pred = predictions[i]
            correct = 1 if (pred > 0 and actual_direction > 0) or (pred < 0 and actual_direction < 0) else 0
            if correct:
                self.alphas[i] += 1
            else:
                self.betas[i] += 1

        total_alpha = np.sum(self.alphas)
        total_beta = np.sum(self.betas)
        if total_alpha + total_beta > 0:
            self.weights = (self.alphas) / (self.alphas + self.betas + 1e-10)
            self.weights = self.weights / (np.sum(self.weights) + 1e-10)

        self.total_predictions += 1

    def predict(self, model_predictions):
        if len(model_predictions) != self.n_models:
            return 0.0, 0.0
        weighted_pred = np.average(model_predictions, weights=self.weights)

        model_votes = np.array([1 if p > 0 else -1 for p in model_predictions])
        weighted_vote = np.dot(self.weights, model_votes)
        confidence = abs(weighted_vote)
        agreement = max(np.mean(model_votes > 0), np.mean(model_votes < 0))

        model_uncertainty = np.std([
            abs(p - weighted_pred) for p in model_predictions
        ])
        uncertainty_penalty = 1 / (1 + model_uncertainty + 1e-10)

        return float(weighted_pred), float(confidence * uncertainty_penalty)

    def get_model_reliability(self):
        return {
            name: {
                "weight": round(float(w), 4),
                "accuracy": round(float(a / (a + b + 1e-10)), 3),
                "samples": int(a + b - 2 * self.alpha_prior + 2),
            }
            for name, w, a, b in zip(self.model_names, self.weights, self.alphas, self.betas)
        }

    def save(self):
        try:
            os.makedirs(os.path.dirname(BMA_PATH), exist_ok=True)
            with open(BMA_PATH, "wb") as f:
                pickle.dump({
                    "weights": self.weights,
                    "alphas": self.alphas,
                    "betas": self.betas,
                    "model_names": self.model_names,
                    "total_predictions": self.total_predictions,
                    "trained_at": datetime.now().isoformat(),
                }, f)
            logger.info(f"BMA saved: {dict(zip(self.model_names, [round(w, 3) for w in self.weights]))}")
            return True
        except Exception as e:
            logger.warning(f"BMA save failed: {e}")
            return False

    def load(self):
        if not os.path.exists(BMA_PATH):
            return False
        try:
            with open(BMA_PATH, "rb") as f:
                data = pickle.load(f)
            self.weights = data["weights"]
            self.alphas = data["alphas"]
            self.betas = data["betas"]
            self.model_names = data["model_names"]
            self.total_predictions = data["total_predictions"]
            self.n_models = len(self.weights)
            self.is_fitted = True
            logger.info(f"BMA loaded: {dict(zip(self.model_names, [round(w, 3) for w in self.weights]))}")
            return True
        except Exception as e:
            logger.warning(f"BMA load failed: {e}")
            return False

    def reset(self):
        self.weights = np.ones(self.n_models) / self.n_models
        self.alphas = np.ones(self.n_models) * self.alpha_prior
        self.betas = np.ones(self.n_models) * self.alpha_prior
        self.total_predictions = 0

    def uncertainty_interval(self, model_predictions, ci=0.95):
        weighted = np.average(model_predictions, weights=self.weights)
        std = np.std([
            abs(p - weighted) for p in model_predictions
        ])
        z = 1.96 if ci == 0.95 else 1.645
        return float(weighted - z * std / np.sqrt(self.n_models)), float(weighted + z * std / np.sqrt(self.n_models))


bma_ensemble = BayesianModelAveraging()
bma_ensemble.load()