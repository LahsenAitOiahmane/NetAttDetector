# models/pca_svm.py

import logging
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from models.base_model import BaseModel

class PCASVM(BaseModel):
    def __init__(self):
        super().__init__("PCA + SVM")
        self.model = None

    def train(self, X_train, y_train):
        try:
            pca = PCA(n_components=20, whiten=True, random_state=42)
            svm = SVC(kernel='rbf', class_weight='balanced', C=1, gamma=0.01)
            self.model = make_pipeline(pca, svm)
            self.model.fit(X_train, y_train)
            logging.info('PCA + SVM trained successfully.')
        except Exception as e:
            logging.error(f"Error training PCA + SVM: {e}")
            raise

    def predict(self, X_val):
        try:
            return self.model.predict(X_val)
        except Exception as e:
            logging.error(f"Error predicting with PCA + SVM: {e}")
            raise
