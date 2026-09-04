import numpy as np
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import (Dense, Dropout, BatchNormalization, Input, LayerNormalization,
                          MultiHeadAttention, GlobalAveragePooling1D, Add, Flatten, Concatenate)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


class MyTransformer:
    def __init__(self, args):
        self.model = None
        self.is_model_created = False
        self.d_model = getattr(args, 'd_model', 64)
        self.num_heads = getattr(args, 'num_heads', 4)
        self.ff_dim = getattr(args, 'ff_dim', 128)
        self.num_transformer_blocks = getattr(args, 'num_transformer_blocks', 3)
        self.dropout_rate = getattr(args, 'dropout_rate', 0.2)
        self.epochs = getattr(args, 'epochs', 100)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.lookback = getattr(args, 'lookback', 20)
        self.sc_in = MinMaxScaler(feature_range=(0, 1))
        self.sc_out = MinMaxScaler(feature_range=(0, 1))

    def _transformer_block(self, inputs, d_model, num_heads, ff_dim, dropout_rate):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(inputs, inputs)
        attn = Dropout(dropout_rate)(attn)
        out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attn]))
        ffn = Dense(ff_dim, activation='relu')(out1)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        return LayerNormalization(epsilon=1e-6)(Add()([out1, ffn]))

    def create_model(self, n_features):
        seq_len = self.lookback
        inputs = Input(shape=(seq_len, n_features))
        x = Dense(self.d_model)(inputs)
        x = BatchNormalization()(x)
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_block(x, self.d_model, self.num_heads, self.ff_dim, self.dropout_rate)
        x = GlobalAveragePooling1D()(x)
        x = Dense(self.d_model // 2, activation='relu')(x)
        x = Dropout(self.dropout_rate * 0.5)(x)
        x = Dense(self.d_model // 4, activation='relu')(x)
        outputs = Dense(1)(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss='huber',
            optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
            metrics=['mae']
        )

    def _create_sequences(self, data_x, data_y):
        X, y = [], []
        for i in range(len(data_x) - self.lookback):
            X.append(data_x[i:i + self.lookback])
            y.append(data_y[i + self.lookback])
        return np.array(X), np.array(y)

    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1].astype(float)
        train_y = data_x[:, -1].astype(float)

        train_x_scaled = self.sc_in.fit_transform(train_x)
        train_y = train_y.reshape(-1, 1)
        train_y_scaled = self.sc_out.fit_transform(train_y).flatten()

        X_seq, y_seq = self._create_sequences(train_x_scaled, train_y_scaled)
        if len(X_seq) < 10:
            return

        if not self.is_model_created:
            self.create_model(train_x_scaled.shape[1])
            self.is_model_created = True

        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, min_delta=1e-5),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6)
        ]

        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(X_seq)),
            verbose=0,
            shuffle=False,
            callbacks=callbacks
        )

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        test_x_scaled = self.sc_in.transform(test_x)

        preds = []
        for i in range(len(test_x_scaled)):
            if i < self.lookback:
                pad = np.tile(test_x_scaled[0], (self.lookback - i - 1, 1))
                seq = np.vstack([pad, test_x_scaled[:i + 1]])
            else:
                seq = test_x_scaled[i - self.lookback + 1:i + 1]
            seq = seq.reshape(1, self.lookback, -1)
            p = self.model.predict(seq, verbose=0)[0, 0]
            preds.append(p)

        preds = np.array(preds).reshape(-1, 1)
        return self.sc_out.inverse_transform(preds).flatten()