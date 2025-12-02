# models/one_class_svm.py

import logging
from sklearn.svm import OneClassSVM
from models.base_model import BaseModel

class OneClassSVMModel(BaseModel):
    def __init__(self):
        super().__init__("One-Class SVM")
        self.model = None

    def train(self, X_train, y_train=None):
        try:
            self.model = OneClassSVM(kernel='rbf', gamma=0.01)
            self.model.fit(X_train)
            logging.info('One-Class SVM trained successfully.')
        except Exception as e:
            logging.error(f"Error training One-Class SVM: {e}")
            raise

    def predict(self, X_val):
        try:
            predictions = self.model.predict(X_val)
            return ['Normal' if p == 1 else 'Anomaly' for p in predictions]
        except Exception as e:
            logging.error(f"Error predicting with One-Class SVM: {e}")
            raise
