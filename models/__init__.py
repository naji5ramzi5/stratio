from .random_forest import RandomForest
from .sarimax import Sarimax
from .orbit import Orbit
from .arima import MyARIMA
from .prophet import MyProphet
from .xgboost import MyXGboost
from .ensemble import MyEnsemble

try:
    from .LSTM import MyLSTM
except ImportError:
    MyLSTM = None

try:
    from .GRU import MyGRU
except ImportError:
    MyGRU = None

try:
    from .neural_prophet import Neural_Prophet
except ImportError:
    Neural_Prophet = None


MODELS = {'random_forest': RandomForest,
          'sarimax': Sarimax,
          'orbit': Orbit,
          'arima': MyARIMA,
          'prophet': MyProphet,
          'xgboost': MyXGboost,
          'ensemble': MyEnsemble
          }

if MyLSTM is not None:
    MODELS['lstm'] = MyLSTM
if MyGRU is not None:
    MODELS['gru'] = MyGRU
if Neural_Prophet is not None:
    MODELS['neural_prophet'] = Neural_Prophet


