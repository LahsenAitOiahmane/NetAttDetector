# main.py

import logging
import joblib
import os
from data.data_loader import DataLoader
from models.bagging_random_forest import BaggingRandomForest
from models.xgboost import XGBoostModel
from models.pca_svm import PCASVM
# from models.isolation_forest import IsolationForestModel
# from models.one_class_svm import OneClassSVMModel
from models.extra_trees import ExtraTreesModel
from models.mlp_neural_network import MLPNeuralNetwork
from utils.logging_setup import setup_logging

# Directory to save models
MODEL_SAVE_DIR = 'saved_models/'

def save_model(model, model_name):
    """Save the trained model to disk."""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    logging.info(f"Saved {model_name} model to {model_path}")

def main():
    setup_logging()

    data_loader = DataLoader('assets/train_net.csv', 'assets/test_net.csv')
    try:
        # Load and preprocess the data
        X_train, X_val, y_train, y_val = data_loader.load_and_preprocess()
        print(f"Shapes of processed data:\nX_train: {X_train.shape}\nX_val: {X_val.shape}\ny_train: {y_train.shape}\ny_val: {y_val.shape}")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")
        return  # Exit if data loading fails

    models = {
        'Bagging Random Forest': BaggingRandomForest(),
        'XGBoost': XGBoostModel(),
        'PCA + SVM': PCASVM(),
        # 'Isolation Forest': IsolationForestModel(),
        # 'One-Class SVM': OneClassSVMModel(),
        'Extra Trees': ExtraTreesModel(),
        'MLP Neural Network': MLPNeuralNetwork()
    }

    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        try:
            # Train the model
            model.train(X_train, y_train)
            # Predict and evaluate
            y_pred = model.predict(X_val)
            model.evaluate(y_val, y_pred)
            # Save the trained model
            save_model(model, model_name)
        except Exception as e:
            logging.error(f"Error with {model_name}: {e}")

if __name__ == "__main__":
    main()
