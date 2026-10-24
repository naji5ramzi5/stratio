import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, prices, volumes=None):
        features = self._build_features(prices, volumes)
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(features_scaled)
        self.is_fitted = True

    def _build_features(self, prices, volumes=None):
        p = np.asarray(prices, dtype=float)
        features = []
        returns = np.diff(p) / p[:-1]
        for i in range(len(p)):
            feat = [p[i]]
            if i >= 1:
                feat.append(returns[i - 1])
                feat.append(abs(returns[i - 1]))
            else:
                feat.extend([0, 0])
            if i >= 5:
                feat.append(np.std(returns[max(0, i - 5):i]))
                feat.append(np.mean(returns[max(0, i - 5):i]))
            else:
                feat.extend([0, 0])
            if i >= 20:
                feat.append(np.std(returns[max(0, i - 20):i]))
            else:
                feat.append(0)
            if volumes is not None and i < len(volumes):
                v = float(volumes[i])
                feat.append(np.log(v + 1))
                if i >= 5:
                    v_avg = np.mean([float(volumes[max(0, i - 5):i + 1])])
                    feat.append(v / (v_avg + 1e-10))
                else:
                    feat.append(1)
            else:
                feat.extend([0, 1])
            features.append(feat)
        return np.array(features)

    def predict(self, prices, volumes=None):
        if not self.is_fitted or self.model is None:
            return np.zeros(len(prices))
        features = self._build_features(prices, volumes)
        features_scaled = self.scaler.transform(features)
        scores = self.model.decision_function(features_scaled)
        labels = self.model.predict(features_scaled)
        return labels, scores


_regime_cache = {}


def detect_regime_change(prices, window_short=20, window_long=100):
    p = np.asarray(prices, dtype=float)
    if len(p) < window_long:
        return "unknown", 0

    returns = np.diff(p) / p[:-1]
    short_vol = np.std(returns[-window_short:]) * np.sqrt(365)
    long_vol = np.std(returns[-window_long:]) * np.sqrt(365)
    vol_regime = short_vol / (long_vol + 1e-10)

    sma_short = np.mean(p[-window_short:])
    sma_long = np.mean(p[-window_long:])
    trend = (sma_short - sma_long) / (sma_long + 1e-10) * 100

    if vol_regime > 1.5 and abs(trend) > 2:
        regime = "high_volatility_trend"
    elif vol_regime > 1.5:
        regime = "high_volatility"
    elif vol_regime < 0.7 and abs(trend) < 1:
        regime = "low_volatility_ranging"
    elif vol_regime < 0.7 and abs(trend) > 3:
        regime = "low_volatility_trend"
    elif trend > 3:
        regime = "uptrend"
    elif trend < -3:
        regime = "downtrend"
    else:
        regime = "ranging"

    return regime, round(trend, 2)


def detect_flash_crash(prices, threshold=0.05):
    p = np.asarray(prices, dtype=float)
    if len(p) < 10:
        return False, 0
    recent = p[-10:]
    drops = (recent[:-1] - recent[1:]) / recent[:-1]
    max_drop = np.max(drops)
    return max_drop > threshold, round(max_drop * 100, 2)


def volatility_regime(prices, periods=[7, 14, 30]):
    p = np.asarray(prices, dtype=float)
    if len(p) < max(periods) + 1:
        return "insufficient_data", 0
    returns = np.diff(p) / p[:-1]
    vols = {}
    for period in periods:
        vols[f"vol_{period}"] = float(np.std(returns[-period:]) * np.sqrt(365) * 100)
    avg_vol = np.mean(list(vols.values()))
    if avg_vol > 80:
        label = "extreme"
    elif avg_vol > 50:
        label = "high"
    elif avg_vol > 30:
        label = "moderate"
    else:
        label = "low"
    return label, round(avg_vol, 2)