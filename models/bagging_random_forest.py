# models/bagging_random_forest.py

import logging
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from models.base_model import BaseModel

class BaggingRandomForest(BaseModel):
    def __init__(self):
        super().__init__("Bagging Random Forest")
        self.model = None

    def train(self, X_train, y_train):
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model = BaggingClassifier(estimator=rf, n_estimators=10, random_state=42)
            self.model.fit(X_train, y_train)
            logging.info('Bagging Random Forest trained successfully.')
        except Exception as e:
            logging.error(f"Error training Bagging Random Forest: {e}")
            raise

    def predict(self, X_val):
        try:
            return self.model.predict(X_val)
        except Exception as e:
            logging.error(f"Error predicting with Bagging Random Forest: {e}")
            raise
