import pandas as pd

def load_data(file_path):
  """Loads and preprocesses the cricket data."""
  df = pd.read_csv(file_path)

  # Convert categorical variables to numerical if necessary:
  df = pd.get_dummies(df, columns=['shot_type', 'line_length','pitch_condition', 'weather_condition'])
  # Handle NaN values in other relevant numerical columns
  numerical_columns_to_fillna = ["pace","spin","movement"]
  df[numerical_columns_to_fillna] = df[numerical_columns_to_fillna].fillna(0)

def create_rolling_features(df, window_size=5):
  """Creates rolling average features for the numerical columns"""
  numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

  for col in numerical_cols:
      if col != 'runs' and col != 'ball_number':  # Avoid rolling on the 'runs' and 'ball number' since they will be the target and the index of the data points
        df[f'{col}_rolling_avg_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).mean()

  # One hot encoding for teams and player columns
  df = pd.get_dummies(df, columns=['batting_team', 'bowling_team','striker', 'non_striker','bowler'])

  return df