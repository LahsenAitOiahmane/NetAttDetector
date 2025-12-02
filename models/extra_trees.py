# models/extra_trees.py

import logging
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from models.base_model import BaseModel

class ExtraTreesModel(BaseModel):
    def __init__(self):
        super().__init__("Extra Trees")
        self.model = None

    def train(self, X_train, y_train):
        try:
            parameters = {'n_estimators': [50, 100, 150]}
            grid_search = GridSearchCV(ExtraTreesClassifier(), parameters, scoring='accuracy', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = ExtraTreesClassifier(n_estimators=grid_search.best_params_['n_estimators'])
            self.model.fit(X_train, y_train)
            logging.info(f"Best Extra Trees params: {grid_search.best_params_}")
            logging.info(f"Best Extra Trees score: {grid_search.best_score_}")
            logging.info('Extra Trees trained successfully.')
        except Exception as e:
            logging.error(f"Error training Extra Trees: {e}")
            raise

    def predict(self, X_val):
        try:
            return self.model.predict(X_val)
        except Exception as e:
            logging.error(f"Error predicting with Extra Trees: {e}")
            raise
