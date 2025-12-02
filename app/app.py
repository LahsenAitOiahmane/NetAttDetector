# app.py
import sys
import os
import logging
import joblib
import csv
import pyshark
import pandas as pd
from datetime import datetime

# Adjust the Python path to include the root directory and models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the correct directories
from models.bagging_random_forest import BaggingRandomForest
from models.xgboost import XGBoostModel
from models.pca_svm import PCASVM
from models.extra_trees import ExtraTreesModel
from models.mlp_neural_network import MLPNeuralNetwork
from utils.logging_setup import setup_logging

# Directory to save models
MODEL_SAVE_DIR = 'saved_models/'

# Define the full list of features expected by the models
EXPECTED_FEATURES = [
    'IN_BYTES', 'ANOMALY', 'TCP_WIN_MSS_IN', 'L4_DST_PORT', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN',
    'OUT_BYTES', 'FIRST_SWITCHED', 'TOTAL_FLOWS_EXP', 'FLOW_DURATION_MILLISECONDS', 
    'LAST_SWITCHED', 'L4_SRC_PORT', 'TCP_FLAGS', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MAX_OUT', 
    'IN_PKTS', 'PROTOCOL', 'SRC_TOS', 'OUT_PKTS', 'TCP_WIN_MIN_OUT', 'TCP_WIN_SCALE_OUT', 'DST_TOS'
]

def save_model(model, model_name):
    """Save the trained model to disk."""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    logging.info(f"Saved {model_name} model to {model_path}")

def load_models():
    """Load all models from the saved models directory."""
    models = {}
    for model_name in ['Bagging Random Forest', 'XGBoost', 'PCA + SVM', 'Extra Trees', 'MLP Neural Network']:
        model_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}.joblib')
        models[model_name] = joblib.load(model_path)
    logging.info("All models loaded successfully.")
    return models

def extract_features(packet):
    def safe_int_conversion(value):
        """Convert a value to an integer safely with specified base, defaulting to 10."""
        try:
            # Handle hexadecimal strings
            if isinstance(value, str) and value.startswith('0x'):
                return int(value, 16)
            return int(value)
        except (ValueError, TypeError):
            return 0  # Return a default value (0) on conversion failure

    # Extract features from Pyshark packet, handling missing values and complex calculations
    features = {
        'IN_BYTES': safe_int_conversion(packet.length) if hasattr(packet, 'length') else 0,  # Packet length in bytes
        'ANOMALY': 0,  # Placeholder for anomaly detection
        'TCP_WIN_MSS_IN': safe_int_conversion(packet.tcp.options_mss) if hasattr(packet, 'tcp') and 'options_mss' in packet.tcp.field_names else 0,
        'L4_DST_PORT': safe_int_conversion(packet.tcp.dstport) if hasattr(packet, 'tcp') else 0,
        'TCP_WIN_MAX_IN': safe_int_conversion(packet.tcp.window_size) if hasattr(packet, 'tcp') else 0,
        'TCP_WIN_MIN_IN': safe_int_conversion(packet.tcp.window_size) if hasattr(packet, 'tcp') else 0,  # Simplified
        'OUT_BYTES': 0,  # Placeholder, needs bidirectional capture logic
        'FIRST_SWITCHED': datetime.now().timestamp(),  # Use current timestamp as placeholder
        'TOTAL_FLOWS_EXP': 1,  # Placeholder, replace with actual logic
        'FLOW_DURATION_MILLISECONDS': 0,  # Placeholder, calculate if possible
        'LAST_SWITCHED': datetime.now().timestamp(),  # Use current timestamp as placeholder
        'L4_SRC_PORT': safe_int_conversion(packet.tcp.srcport) if hasattr(packet, 'tcp') else 0,
        'TCP_FLAGS': safe_int_conversion(packet.tcp.flags) if hasattr(packet, 'tcp') else 0,  # Safe int conversion
        'TCP_WIN_SCALE_IN': safe_int_conversion(packet.tcp.options_wscale) if hasattr(packet, 'tcp') and 'options_wscale' in packet.tcp.field_names else 0,
        'TCP_WIN_MAX_OUT': 0,  # Placeholder for outbound, needs tracking logic
        'IN_PKTS': 1,  # Count each packet as 1
        'PROTOCOL': safe_int_conversion(packet.ip.proto) if hasattr(packet, 'ip') else 0,
        'SRC_TOS': safe_int_conversion(packet.ip.dsfield) if hasattr(packet, 'ip') else 0,
        'OUT_PKTS': 0,  # Placeholder for outbound packet tracking
        'TCP_WIN_MIN_OUT': 0,  # Placeholder
        'TCP_WIN_SCALE_OUT': 0,  # Placeholder
        'DST_TOS': 0,  # Placeholder for destination TOS
    }

    # Ensure all expected features are present and in the correct order
    aligned_features = {feature: features.get(feature, 0) for feature in EXPECTED_FEATURES}
    return aligned_features


def predict_and_log(packet_id, features, models, csv_writer):
    # Convert features to DataFrame with correct order
    features_df = pd.DataFrame([features])[EXPECTED_FEATURES]
    
    # Log prediction results for each model
    for model_name, model in models.items():
        try:
            prediction = model.predict(features_df)
            logging.info(f"Packet ID: {packet_id}, Model: {model_name}, Prediction: {prediction[0]}")
            
            # Save to CSV for model updates
            csv_writer.writerow([packet_id, model_name, prediction[0]] + list(features.values()))
        except Exception as e:
            logging.error(f"Error predicting with {model_name}: {e}")

def main():
    setup_logging()

    # Load pre-trained models
    models = load_models()

    # Define the expected header for CSV based on the feature extraction
    csv_header = ['Packet_ID', 'Model', 'Prediction'] + EXPECTED_FEATURES

    # Prepare CSV for storing packet information and predictions
    csv_filename = f'captured_packets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(csv_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write header directly
        csv_writer.writerow(csv_header)

        packet_counter = 1  # Initialize packet counter

        # Start capturing network traffic using Pyshark
        capture = pyshark.LiveCapture(interface='Wi-Fi')  # Replace 'Wi-Fi' with your network interface name
        logging.info("Started capturing network traffic.")

        for packet in capture.sniff_continuously():
            try:
                features = extract_features(packet)
                predict_and_log(packet_counter, features, models, csv_writer)
                packet_counter += 1
            except Exception as e:
                logging.error(f"Error processing packet ID {packet_counter}: {e}")

if __name__ == "__main__":
    main()
