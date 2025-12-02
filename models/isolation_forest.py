# models/isolation_forest.py

import logging
from sklearn.ensemble import IsolationForest
from models.base_model import BaseModel

class IsolationForestModel(BaseModel):
    def __init__(self):
        super().__init__("Isolation Forest")
        self.model = None

    def train(self, X_train, y_train=None):
        try:
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model.fit(X_train)
            logging.info('Isolation Forest trained successfully.')
        except Exception as e:
            logging.error(f"Error training Isolation Forest: {e}")
            raise

    def predict(self, X_val):
        try:
            predictions = self.model.predict(X_val)
            return ['Normal' if p == 1 else 'Anomaly' for p in predictions]
        except Exception as e:
            logging.error(f"Error predicting with Isolation Forest: {e}")
            raise
