import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import joblib
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c * 1000  # Convert to meters
    return distance

# Load and preprocess data (original function)
def load_data():
    try:
        if not os.path.exists('pgh_train.csv'):
            raise FileNotFoundError("pgh_train.csv not found in the current directory")
        traffic_data = pd.read_csv('pgh_train.csv')
        logger.info("Dataset loaded successfully")
        required_columns = ['speed', 'time', 'is.weekday', 'from.x', 'from.y', 'to.x', 'to.y']
        missing_cols = [col for col in required_columns if col not in traffic_data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in dataset: {missing_cols}")
        traffic_data = traffic_data.dropna()
        if traffic_data.empty:
            raise ValueError("Dataset is empty after dropping missing values")
        traffic_data['is.weekday'] = traffic_data['is.weekday'].astype(int)
        traffic_data['segment_length'] = traffic_data.apply(
            lambda row: haversine(row['from.x'], row['from.y'], row['to.x'], row['to.y']), axis=1
        )
        traffic_data['hour'] = traffic_data['time'] % 24
        traffic_data['is_night'] = traffic_data['hour'].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
        traffic_data['is_rush_hour'] = traffic_data['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 16 <= x <= 18) else 0)
        traffic_data['is_midday'] = traffic_data['hour'].apply(lambda x: 1 if 10 <= x <= 15 else 0)
        traffic_data['lon_diff'] = traffic_data['to.x'] - traffic_data['from.x']
        traffic_data['lat_diff'] = traffic_data['to.y'] - traffic_data['from.y']
        traffic_data['dist_to_center'] = traffic_data.apply(
            lambda row: haversine(row['from.x'], row['from.y'], -80.0, 40.44), axis=1
        )
        traffic_data['hour_weekday_interaction'] = traffic_data['hour'] * traffic_data['is.weekday']
        traffic_data['hour_rush_interaction'] = traffic_data['hour'] * traffic_data['is_rush_hour']
        traffic_data['length_dist_interaction'] = traffic_data['segment_length'] * traffic_data['dist_to_center']
        traffic_data['speed_trend'] = traffic_data['time'] / 24
        traffic_data['lon_lat_interaction'] = traffic_data['lon_diff'] * traffic_data['lat_diff']
        traffic_data['hour_sin'] = np.sin(2 * np.pi * traffic_data['hour'] / 24)
        traffic_data['hour_cos'] = np.cos(2 * np.pi * traffic_data['hour'] / 24)
        traffic_data['day_sin'] = np.sin(2 * np.pi * traffic_data['time'] / (24 * 7))
        traffic_data['day_cos'] = np.cos(2 * np.pi * traffic_data['time'] / (24 * 7))
        for col in ['segment_length', 'dist_to_center']:
            traffic_data[f'{col}_squared'] = traffic_data[col] ** 2
        segment_speeds = traffic_data.groupby(['from.x', 'from.y', 'to.x', 'to.y'])['speed'].agg(['mean', 'std']).reset_index()
        segment_speeds.columns = ['from.x', 'from.y', 'to.x', 'to.y', 'avg_segment_speed', 'std_segment_speed']
        traffic_data = traffic_data.merge(segment_speeds, on=['from.x', 'from.y', 'to.x', 'to.y'], how='left')
        traffic_data['avg_segment_speed'] = traffic_data['avg_segment_speed'].fillna(traffic_data['avg_segment_speed'].mean())
        traffic_data['std_segment_speed'] = traffic_data['std_segment_speed'].fillna(traffic_data['std_segment_speed'].mean())
        segment_time_speeds = traffic_data.groupby(['from.x', 'from.y', 'to.x', 'to.y', 'hour', 'is.weekday'])['speed'].mean().reset_index()
        segment_time_speeds.columns = ['from.x', 'from.y', 'to.x', 'to.y', 'hour', 'is.weekday', 'avg_speed_by_time']
        traffic_data = traffic_data.merge(segment_time_speeds, on=['from.x', 'from.y', 'to.x', 'to.y', 'hour', 'is.weekday'], how='left')
        traffic_data['avg_speed_by_time'] = traffic_data['avg_speed_by_time'].fillna(traffic_data['avg_segment_speed'])
        traffic_data['speed_dist_interaction'] = traffic_data['avg_segment_speed'] * traffic_data['dist_to_center']
        traffic_data['speed_hour_interaction'] = traffic_data['avg_segment_speed'] * traffic_data['hour']
        traffic_data['speed_time_interaction'] = traffic_data['avg_speed_by_time'] * traffic_data['hour']
        traffic_data['speed_rush_interaction'] = traffic_data['avg_segment_speed'] * traffic_data['is_rush_hour']
        coords = traffic_data[['from.x', 'from.y', 'to.x', 'to.y']]
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        traffic_data['cluster'] = kmeans.fit_predict(coords)
        try:
            joblib.dump(kmeans, 'kmeans_model.pkl')
            logger.info("Saved kmeans_model.pkl")
        except Exception as e:
            logger.error(f"Failed to save kmeans_model.pkl: {e}")
            raise
        for col in ['segment_length', 'dist_to_center', 'avg_segment_speed', 'std_segment_speed', 'avg_speed_by_time']:
            traffic_data[f'log_{col}'] = np.log1p(traffic_data[col].clip(lower=0))
            traffic_data[f'log_{col}'] = traffic_data[f'log_{col}'].fillna(traffic_data[f'log_{col}'].mean())
        for col in ['speed', 'avg_segment_speed', 'avg_speed_by_time', 'segment_length', 'dist_to_center']:
            q_low = traffic_data[col].quantile(0.01)
            q_high = traffic_data[col].quantile(0.99)
            traffic_data = traffic_data[(traffic_data[col] >= q_low) & (traffic_data[col] <= q_high)]
        try:
            joblib.dump(traffic_data, 'preprocessed_data.pkl')
            logger.info("Saved preprocessed_data.pkl")
        except Exception as e:
            logger.error(f"Failed to save preprocessed_data.pkl: {e}")
            raise
        return traffic_data
    except FileNotFoundError as e:
        logger.error(f"Error in load_data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        raise

# Prepare features and target (original function)
def prepare_data(data):
    features = [
        'hour', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is.weekday', 'is_night',
        'is_rush_hour', 'is_midday', 'log_segment_length', 'segment_length_squared',
        'lon_diff', 'lat_diff', 'log_dist_to_center', 'dist_to_center_squared',
        'hour_weekday_interaction', 'hour_rush_interaction', 'length_dist_interaction',
        'lon_lat_interaction', 'speed_trend', 'speed_dist_interaction',
        'speed_hour_interaction', 'speed_time_interaction', 'speed_rush_interaction',
        'cluster', 'log_avg_segment_speed', 'log_std_segment_speed', 'log_avg_speed_by_time'
    ]
    try:
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise KeyError(f"Features missing in data: {missing_features}")
        X = data[features]
        y = data['speed']
        return X, y, features
    except Exception as e:
        logger.error(f"Error in prepare_data: {e}")
        raise

# Train and evaluate model (original function)
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{model_name} Results:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score (Test): {r2:.2f}")
        return y_pred, mse, r2
    except Exception as e:
        logger.error(f"Error in train_evaluate_model ({model_name}): {e}")
        raise

# Plot results (original function)
def plot_results(y_test, predictions, model_names):
    try:
        plt.figure(figsize=(12, 3))
        for i, (pred, name) in enumerate(zip(predictions, model_names)):
            plt.subplot(1, len(predictions), i+1)
            plt.scatter(y_test, pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Speed')
            plt.ylabel('Predicted Speed')
            plt.title(f'{name}')
            plt.tight_layout()
        plt.savefig('model_comparison.png')
        logger.info("Saved model_comparison.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error in plot_results: {e}")
        raise

# Function to prepare input for prediction
def prepare_input_for_prediction(time, is_weekday, from_x, from_y, to_x, to_y, data, scaler, kmeans, selected_features):
    input_data = pd.DataFrame({
        'time': [time],
        'is.weekday': [int(is_weekday)],
        'from.x': [from_x],
        'from.y': [from_y],
        'to.x': [to_x],
        'to.y': [to_y]
    })
    input_data['segment_length'] = input_data.apply(
        lambda row: haversine(row['from.x'], row['from.y'], row['to.x'], row['to.y']), axis=1
    )
    input_data['hour'] = input_data['time'] % 24
    input_data['is_night'] = input_data['hour'].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
    input_data['is_rush_hour'] = input_data['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 16 <= x <= 18) else 0)
    input_data['is_midday'] = input_data['hour'].apply(lambda x: 1 if 10 <= x <= 15 else 0)
    input_data['lon_diff'] = input_data['to.x'] - input_data['from.x']
    input_data['lat_diff'] = input_data['to.y'] - input_data['from.y']
    input_data['dist_to_center'] = input_data.apply(
        lambda row: haversine(row['from.x'], row['from.y'], -80.0, 40.44), axis=1
    )
    input_data['hour_weekday_interaction'] = input_data['hour'] * input_data['is.weekday']
    input_data['hour_rush_interaction'] = input_data['hour'] * input_data['is_rush_hour']
    input_data['length_dist_interaction'] = input_data['segment_length'] * input_data['dist_to_center']
    input_data['speed_trend'] = input_data['time'] / 24
    input_data['lon_lat_interaction'] = input_data['lon_diff'] * input_data['lat_diff']
    input_data['hour_sin'] = np.sin(2 * np.pi * input_data['hour'] / 24)
    input_data['hour_cos'] = np.cos(2 * np.pi * input_data['hour'] / 24)
    input_data['day_sin'] = np.sin(2 * np.pi * input_data['time'] / (24 * 7))
    input_data['day_cos'] = np.cos(2 * np.pi * input_data['time'] / (24 * 7))
    for col in ['segment_length', 'dist_to_center']:
        input_data[f'{col}_squared'] = input_data[col] ** 2
    # Use mean values from training data for segment statistics
    input_data['avg_segment_speed'] = data['avg_segment_speed'].mean()
    input_data['std_segment_speed'] = data['std_segment_speed'].mean()
    input_data['avg_speed_by_time'] = data['avg_speed_by_time'].mean()
    input_data['speed_dist_interaction'] = input_data['avg_segment_speed'] * input_data['dist_to_center']
    input_data['speed_hour_interaction'] = input_data['avg_segment_speed'] * input_data['hour']
    input_data['speed_time_interaction'] = input_data['avg_speed_by_time'] * input_data['hour']
    input_data['speed_rush_interaction'] = input_data['avg_segment_speed'] * input_data['is_rush_hour']
    coords = input_data[['from.x', 'from.y', 'to.x', 'to.y']]
    input_data['cluster'] = kmeans.predict(coords)
    for col in ['segment_length', 'dist_to_center', 'avg_segment_speed', 'std_segment_speed', 'avg_speed_by_time']:
        input_data[f'log_{col}'] = np.log1p(input_data[col].clip(lower=0))
        input_data[f'log_{col}'] = input_data[f'log_{col}'].fillna(input_data[f'log_{col}'].mean())
    features = [
        'hour', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is.weekday', 'is_night',
        'is_rush_hour', 'is_midday', 'log_segment_length', 'segment_length_squared',
        'lon_diff', 'lat_diff', 'log_dist_to_center', 'dist_to_center_squared',
        'hour_weekday_interaction', 'hour_rush_interaction', 'length_dist_interaction',
        'lon_lat_interaction', 'speed_trend', 'speed_dist_interaction',
        'speed_hour_interaction', 'speed_time_interaction', 'speed_rush_interaction',
        'cluster', 'log_avg_segment_speed', 'log_std_segment_speed', 'log_avg_speed_by_time'
    ]
    input_data = input_data[features]
    input_scaled = scaler.transform(input_data)
    input_selected = input_scaled[:, [i for i, f in enumerate(features) if f in selected_features]]
    return input_selected

# Streamlit app
def main_app():
    st.title("Traffic Speed Prediction App")
    st.write("This app predicts traffic speed using various machine learning models and displays model performance comparisons.")

    # Check if models and data are already processed
    model_files = {
        "Polynomial Regression": "Polynomial_Regression_model.pkl",
        "Random Forest": "Random_Forest_model.pkl",
        "Gradient Boosting": "Gradient_Boosting_model.pkl",
        "XGBoost": "XGBoost_model.pkl",
        "Neural Network": "Neural_Network_model.pkl",
        "Stacking Ensemble": "Stacking_Ensemble_model.pkl"
    }
    required_files = ['scaler.pkl', 'selected_features.pkl', 'kmeans_model.pkl', 'preprocessed_data.pkl'] + list(model_files.values())
    all_files_exist = all(os.path.exists(f) for f in required_files)

    if all_files_exist:
        st.write("Loading pre-trained models and data...")
        data = joblib.load('preprocessed_data.pkl')
        scaler = joblib.load('scaler.pkl')
        selected_features = joblib.load('selected_features.pkl')
        kmeans = joblib.load('kmeans_model.pkl')
        models = [(joblib.load(model_files[name]), name) for name in model_files]
    else:
        st.write("Training models (this may take a while)...")
        try:
            # Define expected features
            expected_features = [
                'hour', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is.weekday', 'is_night',
                'is_rush_hour', 'is_midday', 'log_segment_length', 'segment_length_squared',
                'lon_diff', 'lat_diff', 'log_dist_to_center', 'dist_to_center_squared',
                'hour_weekday_interaction', 'hour_rush_interaction', 'length_dist_interaction',
                'lon_lat_interaction', 'speed_trend', 'speed_dist_interaction',
                'speed_hour_interaction', 'speed_time_interaction', 'speed_rush_interaction',
                'cluster', 'log_avg_segment_speed', 'log_std_segment_speed', 'log_avg_speed_by_time',
                'speed'
            ]
            cache_file = 'preprocessed_data.pkl'
            if os.path.exists(cache_file):
                traffic_data = joblib.load(cache_file)
                logger.info("Loaded preprocessed_data.pkl")
                if all(f in traffic_data.columns for f in expected_features):
                    logger.info("Cached data contains all required features")
                    if not os.path.exists('kmeans_model.pkl'):
                        logger.info("kmeans_model.pkl missing, fitting KMeans...")
                        coords = traffic_data[['from.x', 'from.y', 'to.x', 'to.y']]
                        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                        kmeans.fit(coords)
                        joblib.dump(kmeans, 'kmeans_model.pkl')
                        logger.info("Saved kmeans_model.pkl from cached data")
                    data = traffic_data
                else:
                    logger.info("Cached data missing features, regenerating...")
                    data = load_data()
            else:
                logger.info("No cache found, preprocessing data...")
                data = load_data()
            X, y, features = prepare_data(data)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, 'scaler.pkl')
            logger.info("Saved scaler.pkl")
            selector = SelectFromModel(XGBRegressor(n_estimators=50, random_state=42), threshold="0.1*mean")
            selector.fit(X_scaled, y)
            X_selected = selector.transform(X_scaled)
            selected_features = [features[i] for i in selector.get_support(indices=True)]
            joblib.dump(selected_features, 'selected_features.pkl')
            logger.info(f"Saved selected_features.pkl with features: {selected_features}")
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.15, random_state=42)
            poly_reg = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=50.0))
            rf_param_grid = {
                'n_estimators': [500, 1000, 1500],
                'max_depth': [20, 30, 40],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt', 0.3]
            }
            rf = RandomizedSearchCV(
                RandomForestRegressor(random_state=42),
                rf_param_grid,
                n_iter=10,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            gb_param_grid = {
                'n_estimators': [500, 1000, 1500],
                'learning_rate': [0.01, 0.03, 0.05],
                'max_depth': [5, 7, 9],
                'subsample': [0.8, 0.9],
                'min_samples_split': [2]
            }
            gb = RandomizedSearchCV(
                GradientBoostingRegressor(random_state=42),
                gb_param_grid,
                n_iter=10,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            xgb_param_grid = {
                'n_estimators': [500, 1000, 1500],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.03, 0.05],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_lambda': [10.0, 20.0]
            }
            xgb = RandomizedSearchCV(
                XGBRegressor(random_state=42, tree_method='hist'),
                xgb_param_grid,
                n_iter=10,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            nn_param_grid = {
                'hidden_layer_sizes': [(200, 100), (300, 150), (400, 200)],
                'learning_rate_init': [0.0001, 0.0003],
                'max_iter': [3000],
                'alpha': [0.01, 0.05],
                'batch_size': [16],
                'solver': ['adam']
            }
            nn = RandomizedSearchCV(
                MLPRegressor(random_state=42, early_stopping=True, n_iter_no_change=100),
                nn_param_grid,
                n_iter=10,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            estimators = [
                ('xgb', XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.03, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=1000, max_depth=30, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=1000, max_depth=7, learning_rate=0.03, random_state=42)),
                ('nn', MLPRegressor(hidden_layer_sizes=(300, 150), learning_rate_init=0.0003, max_iter=3000, random_state=42))
            ]
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
                cv=3,
                n_jobs=-1
            )
            models = [
                (poly_reg, "Polynomial Regression"),
                (rf, "Random Forest"),
                (gb, "Gradient Boosting"),
                (xgb, "XGBoost"),
                (nn, "Neural Network"),
                (stacking, "Stacking Ensemble")
            ]
            predictions = []
            results = []
            for model, name in models:
                y_pred, mse, r2 = train_evaluate_model(model, X_train, X_test, y_train, y_test, name)
                predictions.append(y_pred)
                results.append((name, mse, r2))
                joblib.dump(model, f'{name.replace(" ", "_")}_model.pkl')
                logger.info(f"Saved {name.replace(' ', '_')}_model.pkl")
            plot_results(y_test, predictions, [name for _, name in models])
            st.write("Model training completed.")
            kmeans = joblib.load('kmeans_model.pkl')
        except Exception as e:
            st.error(f"Error during model training: {e}")
            return

    # Display model comparison plot
    st.subheader("Model Comparison")
    if os.path.exists('model_comparison.png'):
        st.image('model_comparison.png', caption="Model Predictions vs Actual Speeds", use_container_width=True)
    else:
        st.error("Model comparison plot not found. Please ensure the script has run successfully.")

    # Speed prediction form
    st.subheader("Predict Traffic Speed")
    with st.form("prediction_form"):
        time = st.number_input("Time (hours since epoch)", min_value=0.0, step=0.1, value=0.0)
        is_weekday = st.checkbox("Is Weekday?", value=True)
        from_x = st.number_input("From Longitude (from.x)", value=-80.0)
        from_y = st.number_input("From Latitude (from.y)", value=40.44)
        to_x = st.number_input("To Longitude (to.x)", value=-80.0)
        to_y = st.number_input("To Latitude (to.y)", value=40.44)
        model_choice = st.selectbox("Select Model", list(model_files.keys()))
        submitted = st.form_submit_button("Predict Speed")
        if submitted:
            try:
                input_data = prepare_input_for_prediction(time, is_weekday, from_x, from_y, to_x, to_y, data, scaler, kmeans, selected_features)
                model = joblib.load(model_files[model_choice])
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted Speed: {prediction:.2f} units")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main_app()
