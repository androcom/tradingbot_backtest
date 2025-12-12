import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import joblib
import logging
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback # type: ignore
from sklearn.utils.class_weight import compute_class_weight

class LoggingCallback(Callback):
    def __init__(self, logger):
        self.logger = logger
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 10 == 0:
            self.logger.info(f"   [LSTM] Epoch {epoch+1}: loss={logs.get('loss'):.4f}, acc={logs.get('accuracy'):.4f}")

class HybridLearner:
    def __init__(self, model_dir, logger=None):
        self.xgb_path = os.path.join(model_dir, 'xgb_model.pkl')
        self.lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        self.xgb_model = None
        self.lstm_model = None
        self.logger = logger if logger else logging.getLogger("HybridModel")

    def _build_lstm(self, input_shape):
        params = config.LSTM_PARAMS
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params['units_1'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units_2']),
            Dropout(params['dropout']),
            # [MP Important] Mixed Precision 사용 시 출력층은 반드시 float32여야 함
            Dense(3, activation='softmax', dtype='float32') 
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_flat, y_flat, X_seq, y_seq):
        self.logger.info(">> Training XGBoost...")
        classes = np.unique(y_flat)
        
        # 클래스 불균형 처리 (안전장치 추가)
        if len(classes) >= 3:
            weights = compute_class_weight('balanced', classes=classes, y=y_flat)
            class_weights_dict = dict(zip(classes, weights))
            sample_weights = np.array([class_weights_dict.get(y, 1.0) for y in y_flat])
        else:
            class_weights_dict = None
            sample_weights = None

        self.xgb_model = XGBClassifier(**config.XGB_PARAMS)
        self.xgb_model.fit(X_flat, y_flat, sample_weight=sample_weights)
        joblib.dump(self.xgb_model, self.xgb_path)

        self.logger.info(">> Training LSTM...")
        self.lstm_model = self._build_lstm((X_seq.shape[1], X_seq.shape[2]))
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # [가속화] Config의 대용량 Batch Size 사용
        self.lstm_model.fit(
            X_seq, y_seq, 
            epochs=config.ML_EPOCHS, 
            batch_size=config.ML_BATCH_SIZE, 
            callbacks=[es, LoggingCallback(self.logger)],
            verbose=0,
            class_weight=class_weights_dict
        )
        self.lstm_model.save(self.lstm_path)

    def load(self):
        if os.path.exists(self.xgb_path) and os.path.exists(self.lstm_path):
            self.xgb_model = joblib.load(self.xgb_path)
            self.lstm_model = load_model(self.lstm_path)
            return True
        return False

    def predict_proba(self, X_flat, X_seq):
        if not self.xgb_model or not self.lstm_model:
            if not self.load(): raise Exception("Models not loaded")
        
        xgb_p = self.xgb_model.predict_proba(X_flat)
        # [가속화] 추론 시에도 대용량 배치 사용
        lstm_p = self.lstm_model.predict(X_seq, verbose=0, batch_size=config.ML_BATCH_SIZE)
        return (xgb_p + lstm_p) / 2