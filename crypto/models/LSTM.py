import numpy as np
import pandas as pd

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, BatchNormalization, Activation, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler


class MyLSTM:
    def __init__(self, args):
        self.model = None
        self.is_model_created = False
        self.hidden_dim = getattr(args, 'hidden_dim', 128)
        self.num_layers = getattr(args, 'num_lstm_layers', 3)
        self.dropout_rate = getattr(args, 'dropout_rate', 0.3)
        self.epochs = getattr(args, 'epochs', 100)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.use_bidirectional = getattr(args, 'use_bidirectional', True)
        self.sc_in = MinMaxScaler(feature_range=(0, 1))
        self.sc_out = MinMaxScaler(feature_range=(0, 1))

    def create_model(self, shape_):
        self.model = Sequential()
        for i in range(self.num_layers):
            return_seq = i < self.num_layers - 1
            if i == 0:
                if self.use_bidirectional:
                    self.model.add(Bidirectional(LSTM(self.hidden_dim, return_sequences=return_seq), input_shape=(1, shape_)))
                else:
                    self.model.add(LSTM(self.hidden_dim, return_sequences=return_seq, input_shape=(1, shape_)))
            else:
                if self.use_bidirectional:
                    self.model.add(Bidirectional(LSTM(self.hidden_dim, return_sequences=return_seq)))
                else:
                    self.model.add(LSTM(self.hidden_dim, return_sequences=return_seq))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(self.hidden_dim // 2, activation='relu'))
        self.model.add(Dropout(self.dropout_rate * 0.5))
        self.model.add(Dense(1))
        self.model.compile(loss='huber', optimizer=optimizers.Adam(learning_rate=0.001))

    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1]
        train_y = data_x[:, -1]

        if not self.is_model_created:
            self.create_model(train_x.shape[1])
            self.is_model_created = True

        train_x = self.sc_in.fit_transform(train_x)
        train_y = train_y.reshape(-1, 1)
        train_y = self.sc_out.fit_transform(train_y)
        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))

        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, min_delta=1e-5),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6, min_delta=1e-5)
        ]

        self.model.fit(train_x, train_y, epochs=self.epochs, verbose=0, shuffle=False,
                       batch_size=min(self.batch_size, len(train_x)), callbacks=callbacks)

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        test_x = self.sc_in.transform(test_x)
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        pred_y = self.model.predict(test_x, verbose=0)
        pred_y = pred_y.reshape(-1, 1)
        pred_y = self.sc_out.inverse_transform(pred_y)
        return pred_y

