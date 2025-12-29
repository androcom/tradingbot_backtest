import os
import logging
import joblib
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback # type: ignore
from sklearn.utils.class_weight import compute_class_weight

from core import constants as config

class LoggingCallback(Callback):
    def __init__(self, logger):
        self.logger = logger
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 10 == 0:
            self.logger.info(f"   [LSTM] Epoch {epoch+1}: loss={logs.get('loss'):.4f}, acc={logs.get('accuracy'):.4f}")

class HybridLearner:
    def __init__(self, model_dir, logger=None):
        self.model_dir = model_dir
        self.logger = logger if logger else logging.getLogger("HybridModel")
        
        # 튜닝 시 model_dir가 None일 수 있음
        if self.model_dir:
            self.xgb_path = os.path.join(self.model_dir, 'xgb_model.pkl')
            self.lstm_path = os.path.join(self.model_dir, 'lstm_model.h5')
        else:
            self.xgb_path = None
            self.lstm_path = None

        self.xgb_model = None
        self.lstm_model = None
        
        self.xgb_params = config.XGB_PARAMS.copy()
        self.lstm_params = config.LSTM_PARAMS.copy()
        self.batch_size = config.ML_BATCH_SIZE
        self.epochs = config.ML_EPOCHS

    def build_xgb(self):
        """Tuner 호출용 XGB 빌더"""
        self.xgb_model = XGBClassifier(**self.xgb_params)
        return self.xgb_model

    def build_lstm(self, input_shape):
        """Tuner 및 내부 호출용 LSTM 빌더"""
        params = self.lstm_params
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params.get('units_1', 128), return_sequences=True),
            Dropout(params.get('dropout', 0.1)),
            LSTM(params.get('units_2', 48)),
            Dropout(params.get('dropout', 0.1)),
            Dense(3, activation='softmax', dtype='float32') 
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.lstm_model = model
        return model

    def train(self, X_flat, y_flat, X_seq, y_seq):
        self.logger.info(">> Training XGBoost...")
        classes = np.unique(y_flat)
        
        if len(classes) >= 3:
            weights = compute_class_weight('balanced', classes=classes, y=y_flat)
            class_weights_dict = dict(zip(classes, weights))
            sample_weights = np.array([class_weights_dict.get(y, 1.0) for y in y_flat])
        else:
            class_weights_dict = None
            sample_weights = None

        self.build_xgb()
        self.xgb_model.fit(X_flat, y_flat, sample_weight=sample_weights)
        
        if self.xgb_path:
            joblib.dump(self.xgb_model, self.xgb_path)

        self.logger.info(">> Training LSTM...")
        self.build_lstm((X_seq.shape[1], X_seq.shape[2]))
        
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        self.lstm_model.fit(
            X_seq, y_seq, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            callbacks=[es, LoggingCallback(self.logger)],
            verbose=0,
            class_weight=class_weights_dict
        )
        
        if self.lstm_path:
            self.lstm_model.save(self.lstm_path)

    def load(self):
        if not self.xgb_path or not self.lstm_path:
            return False
        if os.path.exists(self.xgb_path) and os.path.exists(self.lstm_path):
            self.xgb_model = joblib.load(self.xgb_path)
            self.lstm_model = load_model(self.lstm_path)
            return True
        return False

    def predict_proba(self, X_flat, X_seq):
        if not self.xgb_model or not self.lstm_model:
            if not self.load(): 
                raise Exception("Models not loaded or trained")
        
        xgb_p = self.xgb_model.predict_proba(X_flat)
        lstm_p = self.lstm_model.predict(X_seq, verbose=0, batch_size=self.batch_size)
        return (xgb_p + lstm_p) / 2