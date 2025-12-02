import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging


class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.numeric_columns = ['L4_SRC_PORT', 'L4_DST_PORT', 'FLOW_DURATION_MILLISECONDS', 'PROTOCOL', 
                            'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_IN', 'TCP_WIN_MIN_OUT', 
                            'TCP_WIN_MSS_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_SCALE_OUT', 'SRC_TOS', 'DST_TOS', 
                            'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS']
        self.revoked_columns = ['FLOW_ID', 'ID', 'ANALYSIS_TIMESTAMP', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 
                        'PROTOCOL_MAP', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'TOTAL_PKTS_EXP', 'TOTAL_BYTES_EXP']
    def load_and_preprocess(self):
        # Load datasets
        logging.info('Loading datasets...')
        try:
            full_train_df = pd.read_csv(self.train_path)
            full_test_df = pd.read_csv(self.test_path)
            logging.info('Datasets loaded successfully.')
            
            # Sample 10% of the data
            train_df = full_train_df.sample(frac=1, random_state=1)
            test_df = full_test_df.sample(frac=1, random_state=1)
            logging.info('Sampled 10% of the datasets.')
            
            # Print data types for debugging
            logging.info(f"Train DataFrame Data info:\n{train_df.info()}")
            logging.info(f"Test DataFrame Data info:\n{test_df.info()}")
            
            print(full_train_df.shape, full_test_df.shape)
            print(train_df.shape, test_df.shape)

            # # Convert numeric columns to the correct types
            # numeric_columns = self.numeric_columns

            # for col in numeric_columns:
            #     train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            #     test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
            
            # # Handle missing values
            # logging.info('Handling missing values...')
            # train_numeric_df = train_df[numeric_columns]  # Only numeric columns
            # test_numeric_df = test_df[numeric_columns]    # Only numeric columns

            # # Fill missing values in numeric columns
            # train_df[numeric_columns] = train_numeric_df.fillna(train_numeric_df.mean())
            # test_df[numeric_columns] = test_numeric_df.fillna(test_numeric_df.mean())

            # Verify missing values are handled
            train_missing_values_before = train_df.isnull().sum()
            logging.info(f"Missing values in training data before handling:\n{train_missing_values_before}")

            test_missing_values_before = test_df.isnull().sum()
            logging.info(f"Missing values in testing data before handling:\n{test_missing_values_before}")

        except Exception as e:
            logging.error(f"Error loading or processing datasets: {e}")




        # Handle missing values in features and labels
        logging.info('Handling missing values...')

        # Drop rows where 'ALERT' is missing
        train_df = train_df.dropna(subset=['ALERT'])

        # Optionally, fill missing values in 'ANOMALY' with mean or another method
        train_df['ANOMALY'] = train_df['ANOMALY'].fillna(train_df['ANOMALY'].mean())

        # Verify missing values are handled
        missing_values_after = train_df.isnull().sum()
        logging.info(f"Missing values after handling:\n{missing_values_after}")

        # Drop irrelevant columns
        logging.info('Dropping irrelevant columns...')
        revoked_columns = self.revoked_columns
        train_df = train_df.drop(revoked_columns, axis=1)
        logging.info('Irrelevant columns dropped.')

        # Prepare features and labels
        X = train_df.drop('ALERT', axis=1)
        y = train_df['ALERT']

        # Check for NaNs in features and labels before splitting
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            logging.error('Data contains NaN values. Please handle missing values before splitting.')
            print('Features contain NaNs:\n', X.isnull().sum())
            print('Labels contain NaNs:\n', y.isnull().sum())
        else:
            # Split data into training and validation sets
            logging.info('Splitting data into training and validation sets...')
            try:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                logging.info('Data split successfully.')

                # Scale the features
                logging.info('Scaling features...')
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                logging.info('Features scaled successfully.')
                print(X_train_scaled.shape, X_val_scaled.shape)
                return X_train_scaled, X_val_scaled, y_train, y_val
            
            except ValueError as e:
                logging.error(f"Error during data splitting: {e}")


        
