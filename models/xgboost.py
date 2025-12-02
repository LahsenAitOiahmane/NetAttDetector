# models/xgboost.py

import logging
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(self, X_train, y_train):
        try:
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            self.model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(X_train, y_train_encoded)
            logging.info('XGBoost trained successfully.')
        except Exception as e:
            logging.error(f"Error training XGBoost: {e}")
            raise

    def predict(self, X_val):
        try:
            y_pred_encoded = self.model.predict(X_val)
            return self.label_encoder.inverse_transform(y_pred_encoded)
        except Exception as e:
            logging.error(f"Error predicting with XGBoost: {e}")
            raise
