# models/mlp_neural_network.py

import logging
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from models.base_model import BaseModel

class MLPNeuralNetwork(BaseModel):
    def __init__(self):
        super().__init__("MLP Neural Network")
        self.model = None

    def train(self, X_train, y_train):
        try:
            parameters = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd']
            }
            grid_search = GridSearchCV(MLPClassifier(), parameters, scoring='accuracy', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = MLPClassifier(
                hidden_layer_sizes=grid_search.best_params_['hidden_layer_sizes'],
                activation=grid_search.best_params_['activation'],
                solver=grid_search.best_params_['solver']
            )
            self.model.fit(X_train, y_train)
            logging.info(f"Best MLP params: {grid_search.best_params_}")
            logging.info(f"Best MLP score: {grid_search.best_score_}")
            logging.info('MLP Neural Network trained successfully.')
        except Exception as e:
            logging.error(f"Error training MLP Neural Network: {e}")
            raise

    def predict(self, X_val):
        try:
            return self.model.predict(X_val)
        except Exception as e:
            logging.error(f"Error predicting with MLP Neural Network: {e}")
            raise
