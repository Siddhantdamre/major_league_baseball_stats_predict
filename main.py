import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve
from sklearn.ensemble import StackingRegressor
import warnings

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data handler (no changes needed)
def load_data(file_path):
    """Loads and preprocesses the cricket data."""
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=['shot_type', 'line_length', 'pitch_condition', 'weather_condition'])
    numerical_columns_to_fillna = ["pace", "spin", "movement"]
    df[numerical_columns_to_fillna] = df[numerical_columns_to_fillna].fillna(0)
    return df

def create_rolling_features(df, window_size=5):
    """Creates rolling average features for the numerical columns"""
    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

    for col in numerical_cols:
        if col != 'runs' and col != 'ball_number':  # Avoid rolling on the 'runs' and 'ball number' since they will be the target and the index of the data points
            df[f'{col}_rolling_avg_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).mean()

    # One hot encoding for teams and player columns
    df = pd.get_dummies(df, columns=['batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler'])

    return df


# Feature Engineering Enhancements
def create_advanced_features(df):
    """Creates advanced features like interaction terms and recent form."""

    # Interaction terms
    df['striker_bowler_interaction'] = df['striker_avg_runs'] * df['bowler_avg_wickets']
    df['striker_bowler_sr_interaction'] = df['striker_avg_sr'] * df['bowler_avg_economy']

    # Recent form (last 3 matches) - assuming ball number increases with match progression
    for window in [3]:
        df[f'striker_recent_avg_runs_{window}'] = df.groupby('striker')['runs'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'striker_recent_avg_sr_{window}'] = df.groupby('striker')['striker_avg_sr'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'striker_recent_boundary_percent_{window}'] = df.groupby('striker')['striker_avg_boundary_percent'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'striker_recent_middled_percent_{window}'] = df.groupby('striker')['striker_avg_middled_percent'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'striker_recent_edged_percent_{window}'] = df.groupby('striker')['striker_avg_edged_percent'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)

    return df

def target_encode(df, feature, target, reg_strength=300, prior=0):
    """Performs target encoding with regularisation on a categorical feature."""
    global_mean = df[target].mean()
    encoded_map = df.groupby(feature)[target].agg(["mean", "count"])
    encoded_map["smooth"] = (encoded_map["count"] * encoded_map["mean"] + prior * reg_strength) / (encoded_map["count"] + reg_strength)

    return df[feature].map(encoded_map["smooth"])

def apply_target_encode(df, categorical_columns, target, reg_strength=300, prior=0):
    """Applies target encoding to a list of categorical columns"""
    df_copy = df.copy()  # Avoid the copy with view warning
    for col in categorical_columns:
        df_copy[f'{col}_encoded'] = target_encode(df, col, target, reg_strength, prior)
    return df_copy

def robust_feature_selection(df, feature_columns, target, n_features=30):
    """Performs recursive feature elimination for feature selection."""
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(df[feature_columns], df[target])
    selected_features = [feature for feature, selected in zip(feature_columns, rfe.support_) if selected]
    return selected_features

# Model Training (no changes needed)
def train_model(X_train, y_train):
    """Trains the LightGBM model using Grid Search for tuning."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
    }

    param_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 5, 10]
    }

    model = lgb.LGBMRegressor(**params, seed=42, verbose=-1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    return best_model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints metrics."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R-squared: {r2:.2f}')

def evaluate_model_cv(model, df, feature_columns, target):
    """Evaluates the trained model with cross-validation and prints metrics."""
    match_ids = df['match_id'].unique()
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in tscv.split(match_ids):
        train_matches = match_ids[train_index]
        test_matches = match_ids[test_index]

        train_df = df[df['match_id'].isin(train_matches)]
        test_df = df[df['match_id'].isin(test_matches)]

        X_train = train_df[feature_columns]
        y_train = train_df[target]
        X_test = test_df[feature_columns]
        y_test = test_df[target]

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    print(f'Average RMSE: {np.mean(rmse_scores):.2f}')
    print(f'Average MAE: {np.mean(mae_scores):.2f}')
    print(f'Average R-squared: {np.mean(r2_scores):.2f}')


# Advanced Evaluation Techniques
def plot_calibration_curve(model, X_test, y_test):
    """Plots the calibration curve for a regression model."""
    y_pred = model.predict(X_test)
    true_prob, pred_prob = calibration_curve(y_test, y_pred, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(pred_prob, true_prob, marker='o', linestyle='-', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def error_analysis(model, X_test, y_test, features):
   """Investigates where the model makes the largest errors"""
   y_pred = model.predict(X_test)
   error = abs(y_pred - y_test)
   error_df = pd.DataFrame({'error': error, 'predicted': y_pred, 'actual':y_test}, index=X_test.index)
   error_df = error_df.join(X_test)
   sorted_errors = error_df.sort_values(by='error', ascending=False).head(10)
   print("\nTop 10 largest errors:")
   print(sorted_errors)


# Prediction Visualization
def plot_actual_vs_predicted(model, X_test, y_test):
    """Plots the predicted vs actual runs."""
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Runs')
    plt.ylabel('Predicted Runs')
    plt.title('Actual vs Predicted Runs')
    plt.show()

# Feature Importance
def plot_feature_importance(model, feature_names):
    """Plots the feature importance from the model."""
    feature_importance = pd.Series(model.feature_importances_, index=feature_names)
    feature_importance_sorted = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    feature_importance_sorted.plot(kind='bar')
    plt.title('Feature Importance')
    plt.ylabel('Importance Score')
    plt.show()

# Blending Function
def blend_models(models, weights, X_test):
  """Blends predictions of different models."""
  predictions = np.array([model.predict(X_test) for model in models])
  blended_prediction = np.average(predictions, axis=0, weights=weights)
  return blended_prediction


# Model saving and loading
def save_model(model, file_path):
    joblib.dump(model, file_path)
def load_model(file_path):
    return joblib.load(file_path)

# Main function
if __name__ == "__main__":
    file_path = "cricket_data.csv"  # Or any other path

    df = load_data(file_path)
    df = create_rolling_features(df)
    df = create_advanced_features(df)

    # Target encoding for high cardinality features before creating the train/test split
    categorical_features = ['striker', 'non_striker','bowler', 'batting_team', 'bowling_team']
    df = apply_target_encode(df, categorical_features, 'runs', reg_strength = 300)

    # Example feature selection (adapt based on your analysis)
    feature_columns = [col for col in df.columns if col not in ['runs', 'match_id', 'player_out']]
    target = 'runs'

    # Robust feature selection
    selected_features = robust_feature_selection(df, feature_columns, target, n_features=30)
    print("Selected Features:", selected_features)

    X = df[selected_features]
    y = df[target]

    # Time Series split for training and testing
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    evaluate_model_cv(model, df, selected_features, target)

    plot_actual_vs_predicted(model, X_test, y_test)
    plot_feature_importance(model, selected_features)
    plot_calibration_curve(model, X_test, y_test)
    error_analysis(model, X_test, y_test, selected_features)


    # Create second model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.03,
        'num_leaves': 40,
        'max_depth': -1,
        'seed': 100,
        'verbose': -1
    }
    model_2 = lgb.LGBMRegressor(**params)
    model_2.fit(X_train, y_train)


    # Blending
    blended_predictions = blend_models([model,model_2], [0.6, 0.4], X_test)
    rmse_blended = np.sqrt(mean_squared_error(y_test, blended_predictions))
    mae_blended = mean_absolute_error(y_test, blended_predictions)
    r2_blended = r2_score(y_test, blended_predictions)
    print(f'Blended RMSE: {rmse_blended:.2f}')
    print(f'Blended MAE: {mae_blended:.2f}')
    print(f'Blended R-squared: {r2_blended:.2f}')

    # Save the model
    save_model(model, 'cricket_model.joblib')
    loaded_model = load_model('cricket_model.joblib')