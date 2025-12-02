# models/base_model.py

import logging
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_val):
        pass

    def evaluate(self, y_val, y_pred):
        report = classification_report(y_val, y_pred)
        logging.info(f"{self.model_name} Classification Report:\n{report}")
        return report
