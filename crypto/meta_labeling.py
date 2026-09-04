"""
Meta-Labeling System
Train a secondary classifier to predict WHEN the primary prediction will be correct.
Dramatically reduces false signals.
Based on López de Prado's "Advances in Financial Machine Learning".
"""
import numpy as np
import pandas as pd
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

META_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "meta_labeler.pkl")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MetaLabeler:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.threshold = 0.5
        self.precision_at_threshold = 0.0

    def _build_features(self, prediction_result):
        feat = [
            prediction_result.get("confidence", 50) / 100.0,
            prediction_result.get("filters_passed", 0) / 5.0,
            prediction_result.get("rsi", 50) / 100.0,
            abs(prediction_result.get("predicted_change_pct", 0)) / 10.0,
            min(prediction_result.get("volume_ratio", 1), 3) / 3.0,
            prediction_result.get("atr_pct", 2) / 10.0,
            prediction_result.get("divergence_score", 0) / 25.0,
            abs(prediction_result.get("buy_pressure_pct", 50) - 50) / 50.0,
            (prediction_result.get("nearest_support", 0) > 0) * 1.0,
            (prediction_result.get("nearest_resistance", 0) > 0) * 1.0,
            prediction_result.get("historical_accuracy_pct", 50) / 100.0,
        ]

        regime = prediction_result.get("market_regime", "neutral")
        feat.extend([
            1.0 if regime == "bullish" else 0.0,
            1.0 if regime == "bearish" else 0.0,
            1.0 if regime == "strong_bullish" else 0.0,
            1.0 if regime == "strong_bearish" else 0.0,
        ])

        grade = prediction_result.get("grade", "")
        feat.extend([
            1.0 if "STRONG" in grade else 0.0,
            1.0 if "MODERATE" in grade else 0.0,
            1.0 if "SLIGHT" in grade else 0.0,
            1.0 if "NEUTRAL" in grade else 0.0,
        ])

        return np.array(feat, dtype=np.float64)

    def prepare_training_data(self, tracker_data):
        verified = [p for p in tracker_data if p.get("verified") and p.get("actual_change_pct") is not None]
        if len(verified) < 50:
            logger.warning(f"Meta-labeler: only {len(verified)} verified samples, need >= 50")
            return None, None

        X_list, y_list = [], []
        self.feature_names = [
            "confidence", "filters_passed", "rsi", "pred_change_abs", "volume_ratio",
            "atr_pct", "divergence", "buy_pressure_div", "has_support", "has_resistance",
            "hist_accuracy",
            "regime_bullish", "regime_bearish", "regime_strong_bullish", "regime_strong_bearish",
            "grade_strong", "grade_moderate", "grade_slight", "grade_neutral"
        ]

        for p in verified:
            pred = p["predicted_change_pct"]
            actual = p["actual_change_pct"]
            correct = 1 if (pred > 0 and actual > 0) or (pred < 0 and actual < 0) else 0

            dummy_result = {
                "confidence": p.get("confidence", 50),
                "filters_passed": p.get("filters_passed", 2),
                "rsi": p.get("rsi", 50),
                "predicted_change_pct": pred,
                "volume_ratio": p.get("volume_ratio", 1),
                "atr_pct": p.get("atr_pct", 2),
                "divergence_score": p.get("divergence_score", 0),
                "buy_pressure_pct": p.get("buy_pressure_pct", 50),
                "nearest_support": p.get("nearest_support", 0),
                "nearest_resistance": p.get("nearest_resistance", 0),
                "historical_accuracy_pct": p.get("historical_accuracy_pct", 50),
                "market_regime": p.get("market_regime", "neutral"),
                "grade": p.get("grade", "NEUTRAL"),
            }

            X_list.append(self._build_features(dummy_result))
            y_list.append(correct)

        if len(X_list) < 50:
            return None, None

        X = np.array(X_list)
        y = np.array(y_list)

        pos_ratio = y.mean()
        logger.info(f"Meta-labeler data: {len(X)} samples, {pos_ratio:.1%} positive")

        return X, y

    def train(self, tracker_data=None, X=None, y=None):
        if not HAS_SKLEARN:
            logger.warning("Meta-labeler: sklearn not available")
            return False

        if X is None or y is None:
            if tracker_data is None:
                from prediction_tracker import prediction_tracker as pt
                tracker_data = pt.data.get("predictions", [])
            X, y = self.prepare_training_data(tracker_data)

        if X is None or len(X) < 50:
            logger.warning(f"Meta-labeler: insufficient data ({len(X) if X is not None else 0} samples)")
            return False

        tscv = TimeSeriesSplit(n_splits=3)
        best_model = None
        best_f1 = 0

        for name, ModelClass in [
            ("xgb", XGBClassifier if HAS_XGB else None),
            ("rf", RandomForestClassifier),
            ("gbr", GradientBoostingClassifier),
            ("lr", LogisticRegression),
        ]:
            if ModelClass is None:
                continue
            try:
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    X_tr_s = self.scaler.fit_transform(X_tr)
                    X_val_s = self.scaler.transform(X_val)
                    if name == "xgb":
                        model = ModelClass(n_estimators=80, max_depth=4, learning_rate=0.1,
                                           scale_pos_weight=(1 - y_tr.mean()) / max(y_tr.mean(), 0.01),
                                           random_state=42, verbosity=0)
                    elif name == "rf":
                        model = ModelClass(n_estimators=100, max_depth=6, min_samples_leaf=10,
                                           class_weight="balanced", random_state=42)
                    elif name == "gbr":
                        model = ModelClass(n_estimators=80, max_depth=3, learning_rate=0.1,
                                           random_state=42)
                    else:
                        model = ModelClass(class_weight="balanced", random_state=42, max_iter=200)
                    model.fit(X_tr_s, y_tr)
                    pred = model.predict_proba(X_val_s)[:, 1]
                    f1 = f1_score(y_val, pred > 0.5)
                    scores.append(f1)
                avg_f1 = np.mean(scores)
                logger.info(f"  Meta {name}: F1={avg_f1:.4f}")
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_model = (name, model)
            except Exception as e:
                logger.debug(f"  Meta {name} failed: {e}")

        if best_model is None:
            logger.warning("Meta-labeler: no model trained successfully")
            return False

        name, model = best_model
        X_scaled = self.scaler.fit_transform(X)
        model.fit(X_scaled, y)
        self.model = model
        self.is_fitted = True

        pred_proba = model.predict_proba(X_scaled)[:, 1]
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            prec = precision_score(y, pred_proba > thresh, zero_division=0)
            logger.info(f"  Threshold {thresh:.1f}: precision={prec:.3f}")
            if prec >= 0.5:
                self.threshold = thresh
                self.precision_at_threshold = prec
                break

        logger.info(f"Meta-labeler trained: {name}, threshold={self.threshold:.2f}, precision={self.precision_at_threshold:.3f}")
        return True

    def predict(self, prediction_result):
        if not self.is_fitted or self.model is None:
            # No trained meta-model = no information.  Return "no opinion"
            # (reliable=True) instead of (1.0, False), which silently cut every
            # prediction's confidence by 40% in predictor.py.
            return 1.0, True

        try:
            features = self._build_features(prediction_result).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            proba = float(self.model.predict_proba(features_scaled)[0, 1])
            is_reliable = proba >= self.threshold
            return proba, is_reliable
        except Exception as e:
            logger.debug(f"Meta-labeler predict failed: {e}")
            return 0.5, True

    def save(self):
        if not self.is_fitted:
            return False
        try:
            os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
            with open(META_MODEL_PATH, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                    "precision_at_threshold": self.precision_at_threshold,
                    "trained_at": datetime.now().isoformat(),
                }, f)
            logger.info(f"Meta-labeler saved to {META_MODEL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Meta-labeler save failed: {e}")
            return False

    def load(self):
        if not os.path.exists(META_MODEL_PATH):
            return False
        try:
            with open(META_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            self.threshold = data["threshold"]
            self.precision_at_threshold = data["precision_at_threshold"]
            self.is_fitted = True
            logger.info(f"Meta-labeler loaded (precision={self.precision_at_threshold:.3f}, threshold={self.threshold:.2f})")
            return True
        except Exception as e:
            logger.warning(f"Meta-labeler load failed: {e}")
            return False


meta_labeler = MetaLabeler()
meta_labeler.load()