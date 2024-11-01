import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Any
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NBAPointsPredictor:
    def __init__(self, file_path: str):
        """Initialize the NBA Points Prediction system."""
        self.file_path = "/Users/matthewfox/ARIMA/traditional.csv"
        self.data = None
        self.team_data = None
        self.team_models = {}
        self.team_forecasts = {}
        self.evaluation_metrics = {}

    def load_and_prepare_data(self) -> bool:
        """Load and prepare the data for analysis."""
        try:
            if not os.path.exists(self.file_path):
                logging.error(f"File not found: {self.file_path}")
                return False

            self.data = pd.read_csv(self.file_path)
            self.data['date'] = pd.to_datetime(self.data['date'])

            # Aggregate points by team and date
            self.team_data = (self.data.groupby(['date', 'team'])['PTS']
                              .sum()
                              .reset_index()
                              .set_index('date'))

            logging.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            logging.error(f"File not found: {self.file_path}")
            return False
        except pd.errors.EmptyDataError:
            logging.error("No data: The file is empty.")
            return False
        except KeyError as e:
            logging.error(f"Missing key in data: {e}")
            return False
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return False

    def train_team_model(self, team: str = "Lakers", forecast_periods: int = 5) -> Tuple[Any, pd.DataFrame]:
        """Train ARIMA model for a specific team and generate forecasts."""
        try:
            if self.team_data is None:
                logging.error("Team data is not loaded. Please load the data first.")
                return None, None

            team_points = self.team_data[self.team_data['team'] == team]['PTS']

            # Check if there is enough data to train the model
            if len(team_points) < 10:
                logging.warning(f"Not enough data to train model for team {team}.")
                return None, None

            # Configure ARIMA model with simpler parameters for testing purposes
            model = auto_arima(
                team_points,
                seasonal=True,  # Consider seasonality in NBA games
                m=7,           # Weekly seasonality
                start_p=0,
                start_q=0,
                max_p=2,       # Simplified for testing
                max_q=2,       # Simplified for testing
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            # Fit model and generate forecasts
            model.fit(team_points)
            forecast = model.predict(n_periods=forecast_periods)
            forecast_dates = [team_points.index[-1] + timedelta(days=i*7) for i in range(1, forecast_periods + 1)]
            forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast})

            logging.info(f"Model trained successfully for team {team}.")
            return model, forecast_df
        except Exception as e:
            logging.error(f"Error training model for team {team}: {str(e)}")
            return None, None

# Example usage:
if __name__ == "__main__":
    predictor = NBAPointsPredictor("/Users/matthewfox/ARIMA/traditional.csv")
    if predictor.load_and_prepare_data():
        predictor.train_team_model()
