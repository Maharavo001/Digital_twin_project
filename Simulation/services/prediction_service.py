import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import pickle
import os
from xgboost import Booster
import lightgbm as lgb
from prophet.serialize import model_from_json


class ModelManager:
    def __init__(self, models_dir, model_type, model_format,feature_scaler_dir, target_scaler_dir,
                 dict_of_cols_and_FE,seq_length,df_window_path):
        """
        raw_cols_map, feature_cols_map, target_cols_map: dict {model_name: cols}
        """
        self.services = {}
        for fname in os.listdir(models_dir):
            if fname.endswith(model_format):
                model_name = os.path.splitext(fname)[0]
                self.services[model_name] = PredictionService(
                    model_path=os.path.join(models_dir, fname),
                    model_type = model_type,
                    junction_name= model_name,
                    feature_scaler_path=self.find_scaler_file(
                                directory=feature_scaler_dir,
                                model_name=model_name,
                                kind="feature",
                                exts=['.save', '.pkl']),
                    target_scaler_path=self.find_scaler_file(
                                directory=target_scaler_dir,
                                model_name=model_name,
                                kind="target",
                                exts=['.save', '.pkl']),
                    dict_of_config = dict_of_cols_and_FE,
                    feature_cols = dict_of_cols_and_FE[model_name]["features"],
                    target_cols=model_name,
                    seq_length=seq_length,
                    df_window=pd.read_parquet(os.path.join(df_window_path, f'{model_name}.parquet')).iloc[24*900:24*901]
                )
    
    def find_scaler_file(self, directory, model_name, kind="feature", exts=None):
        """
        Search for a scaler file that matches model_name and kind ('feature' or 'target').
        Accepts multiple extensions (default: .pkl and .save).
        Returns the first exact match found.
        """
        if exts is None:
            exts = [".pkl", ".save"]

        for fname in os.listdir(directory):
            # Split fname without extension
            name_only, ext = os.path.splitext(fname)
            if ext.lower() not in exts:
                continue
            # Split name by underscore to get parts
            parts = name_only.split("_")
            if len(parts) >= 2 and parts[0] == model_name and parts[1].lower() == kind.lower():
                return os.path.join(directory, fname)

        raise FileNotFoundError(f"No {kind} scaler file found for model {model_name} in {directory}")
            
    def get_service(self, model_name):
        return self.services.get(model_name)
        
    def run_models(self):
        results = {}
        for model_name, service in self.services.items():
            #if model_name not in df.columns:
            #    continue
            pred = service.predict_next_step()
            
            results[model_name] = pred
        return results
        
    def update_window(self, predictions):
        for model_name, service in self.services.items():
            row = service.df_window.iloc[-1].copy()

            # Normalize key for predictions (remove dashes)
            pred_key = model_name.replace("-", "")

            # Skip if prediction is missing
            if pred_key not in predictions:
                print(f"Warning: prediction for junction '{model_name}' missing. Skipping.")
                continue

            # Update the model’s target column
            row[model_name] = predictions[pred_key]

            # Push updated row into the service
            service.update_window(row)



class PredictionService:
    def __init__(self, model_path,model_type,junction_name, feature_scaler_path, target_scaler_path,       
                dict_of_config, feature_cols, target_cols, seq_length,df_window):
        """
        Initialize with proper data validation
        """
        self.model = self.load_ml_model(model_path,model_type)
        self.model_type = model_type.lower()
        self.junction_name = junction_name
        self.feature_scaler = self._load_scaler(feature_scaler_path)
        self.target_scaler = self._load_scaler(target_scaler_path)
        self.feature_config = dict_of_config

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_length = seq_length

        self.df_window = df_window

    def _load_scaler(self, path):
        ext = os.path.splitext(path)[1].lower()
        
        if ext in [".pkl", ".save"]:
            # Try joblib first
            try:
                return joblib.load(path)
            except Exception:
                # fallback to standard pickle
                with open(path, "rb") as f:
                    return pickle.load(f)
        else:
            raise ValueError(f"Unsupported scaler file extension: {ext}")
            
    def load_ml_model(self,path, model_type):
        """
        Load models based on type.
        Args:
            path (str): file path
            model_type (str): "keras", "prophet", "xgboost", "lightgbm"
        Returns:
            model object
        """
        if model_type.lower() == "keras" or model_type.lower() == "h5":
            return tf.keras.models.load_model(path,custom_objects={'mse': MeanSquaredError})

        elif model_type.lower() == "prophet":
            with open(path, "r") as fin:
                return model_from_json(fin.read())  # Prophet serialized to JSON

        elif model_type.lower() == "xgboost":
            model = Booster()
            model.load_model(path)
            return model

        elif model_type.lower() == "lightgbm":
            return lgb.Booster(model_file=path)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def preprocess_input(self, df_window):
        """
        Prepare complete feature set before scaling
        """
        # First prepare all features
        df_features = self.prepare_data(df_window)
        
        # Ensure we have all expected features
        missing_features = set(self.feature_cols) - set(df_features.columns)
        if missing_features:
            raise ValueError(f"Missing features after preparation: {missing_features}")
            
        # Scale features
        scaled_features = self.feature_scaler.transform(df_features[self.feature_cols])
        return scaled_features

    def update_window(self, new_row_raw):
        """
        Update the sliding window with new data
        """
        # Convert input to DataFrame if needed
        if isinstance(new_row_raw, dict):
            new_row = pd.DataFrame([new_row_raw], columns=self.raw_cols)
        else:
            new_row = new_row_raw
            
        # Append new data
        self.df_window = pd.concat([
            self.df_window, 
            new_row
        ], ignore_index=True)
        
        # Maintain fixed window size
        if len(self.df_window) > self.seq_length:
            self.df_window = self.df_window.iloc[-self.seq_length:]

    def prepare_data(self, df_raw):
        """
        Create features including lags and rolling mean based on self.feature_config.
        If no config, return original df.
        """
        df = df_raw.copy()

        # If no config, return original df
        if not hasattr(self, "feature_config") or not self.feature_config:
            return df
        junction_config = self.feature_config.get(self.junction_name, {})

        # Add lag features
        lags = junction_config.get("lags", [])
        for lag in lags:
            df[f"{self.junction_name}_lag{lag}"] = df[self.junction_name].shift(lag)

        # Add rolling mean features
        rolling_windows = junction_config.get("rolling", [])
        for w in rolling_windows:
            col_name = f"{self.junction_name}_rollmean{w}"
            df[col_name] = df[self.junction_name].rolling(w).mean()

        # Fill NaNs from shift/rolling
        df = df.fillna(0)
        return df

    def predict_next_step(self, df_window=None):
        """
        Generate a single prediction value using self.model_type.
        Handles LSTM, LightGBM/XGBoost, and Prophet correctly.
        """
        window_to_use = df_window if df_window is not None else self.df_window

        if len(window_to_use) < self.seq_length:
            raise ValueError(f"Need {self.seq_length} samples, got {len(window_to_use)}")

        # Prepare features
        features_scaled = self.preprocess_input(window_to_use)  # shape: (seq_length, n_features)

        model_type = self.model_type.lower()

        if model_type in ["lightgbm", "xgboost", "sklearn", "lgbm"]:
            # Tree-based models use only the last row (single timestep)
            X_input = features_scaled[-1, :].reshape(1, -1)
            pred_scaled = self.model.predict(X_input)
            pred_scaled = pred_scaled[0] if hasattr(pred_scaled, "__iter__") else pred_scaled

        elif model_type == "lstm":
            # LSTM uses the full sequence
            X_input = features_scaled.reshape(1, self.seq_length, -1)
            pred_scaled = self.model.predict(X_input)
            pred_scaled = pred_scaled[0, -1] if pred_scaled.ndim > 1 else pred_scaled[0]

        elif model_type == "prophet":
            # Prophet uses the last row
            df_prophet = window_to_use.copy()
            pred_scaled = self.model.predict(df_prophet)[self.target_cols[0]].values[-1]

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Ensure 2D for scaler inverse_transform
        pred_scaled_array = np.array([[pred_scaled]])
        pred = self.target_scaler.inverse_transform(pred_scaled_array)[0, 0]

        return float(pred)


        
    def predict_from_df(self, df, cols_to_keep):
        """
        Accepts a dataframe and keeps only relevant cols before prediction.
        cols_to_keep: list of column names to retain (e.g. ['scenario_id', 'temperature'])
        """
        df_sub = df[cols_to_keep].copy()

        # update window with only the raw signal col (exclude scenario_id)
        raw_col = [c for c in df_sub.columns if c != 'scenario_id'][0]
        self.update_window(df_sub[[raw_col]])

        return self.predict_next_step()
