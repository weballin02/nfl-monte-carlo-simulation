import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from typing import Dict, Tuple, Any
from datetime import timedelta

class NBAPointsPredictor:
    def __init__(self, file_path: str):
        """Initialize the NBA Points Prediction system."""
        self.file_path = file_path
        self.data = None
        self.team_data = None
        self.team_models = {}
        self.team_forecasts = {}
        self.evaluation_metrics = {}
        
    def load_and_prepare_data(self) -> None:
        """Load and prepare the data for analysis."""
        try:
            self.data = pd.read_csv(self.file_path)
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
            
            # Aggregate points by team and date
            self.team_data = (self.data.groupby(['date', 'team'])['PTS']
                            .sum()
                            .reset_index()
                            .set_index('date'))
            
            # Drop rows with NaN values in important columns
            self.team_data = self.team_data.dropna(subset=['PTS'])
            
            print(f"Data loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def train_team_model(self, team: str, forecast_periods: int = 5) -> Tuple[Any, pd.DataFrame]:
        """Train ARIMA model for a specific team and generate forecasts."""
        team_points = self.team_data[self.team_data['team'] == team]['PTS']
        
        # Reset index for ARIMA compatibility
        team_points = team_points.reset_index(drop=True)
        
        # Configure ARIMA model with optimal parameters
        model = auto_arima(
            team_points,
            seasonal=True,  # Consider seasonality in NBA games
            m=7,           # Weekly seasonality
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        # Fit model and generate forecasts
        model.fit(team_points)
        
        # Ensure the index is datetime to avoid type issues
        last_date = self.team_data[self.team_data['team'] == team].index.max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        forecast, conf_int = model.predict(n_periods=forecast_periods, return_conf_int=True)
        
        # Create forecast DataFrame with confidence intervals
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_PTS': forecast,
            'Lower_CI': conf_int[:, 0],
            'Upper_CI': conf_int[:, 1],
            'Team': team
        })
        
        # Drop rows with NaN values in forecast DataFrame
        forecast_df = forecast_df.dropna(subset=['Predicted_PTS', 'Lower_CI', 'Upper_CI'])
        
        # Convert all relevant columns to numeric to avoid type issues
        forecast_df['Predicted_PTS'] = pd.to_numeric(forecast_df['Predicted_PTS'], errors='coerce')
        forecast_df['Lower_CI'] = pd.to_numeric(forecast_df['Lower_CI'], errors='coerce')
        forecast_df['Upper_CI'] = pd.to_numeric(forecast_df['Upper_CI'], errors='coerce')
        
        return model, forecast_df
    
    def evaluate_model(self, team: str, actual: pd.Series, predicted: np.ndarray) -> Dict:
        """Calculate evaluation metrics for the model."""
        # Avoid division by zero in MAPE calculation
        non_zero_actual = actual[actual != 0]
        if len(non_zero_actual) > 0:
            mape = np.mean(np.abs((non_zero_actual - predicted[:len(non_zero_actual)]) / non_zero_actual)) * 100
        else:
            mape = np.nan

        metrics = {
            'MSE': mean_squared_error(actual, predicted),
            'MAE': mean_absolute_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAPE': mape
        }

        self.evaluation_metrics[team] = metrics
        return metrics
    
    def train_single_team(self, team: str, forecast_periods: int = 5) -> None:
        """Train model for a single team for testing purposes."""
        print(f"\nTraining model for {team}")
        model, forecast_df = self.train_team_model(team, forecast_periods)
        
        self.team_models[team] = model
        self.team_forecasts[team] = forecast_df
        
        # Evaluate model
        team_points = self.team_data[self.team_data['team'] == team]['PTS']
        last_n_periods = team_points.tail(forecast_periods)
        if len(last_n_periods) == forecast_periods:
            metrics = self.evaluate_model(team, last_n_periods, forecast_df['Predicted_PTS'].values)
            print(f"Metrics for {team}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")
        else:
            print(f"Not enough data to evaluate the model for team {team}.")
    
    def visualize_team_prediction(self, team: str) -> None:
        """Create visualization for team predictions."""
        if team in self.team_forecasts and not self.team_forecasts[team].empty:
            plt.figure(figsize=(12, 6))

            # Plot historical data
            team_points = self.team_data[self.team_data['team'] == team]['PTS']
            if not team_points.empty:
                plt.plot(team_points.index, team_points.values, label=f'Historical Points', color='blue', alpha=0.6)
            else:
                print(f"No historical data available for {team}.")

            # Plot predictions with confidence intervals
            forecast_df = self.team_forecasts[team]
            
            # Convert Date to datetime if not already
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce')

            # Convert columns to float to avoid type issues during plotting
            forecast_df['Predicted_PTS'] = forecast_df['Predicted_PTS'].astype(float)
            forecast_df['Lower_CI'] = forecast_df['Lower_CI'].astype(float)
            forecast_df['Upper_CI'] = forecast_df['Upper_CI'].astype(float)
            
            # Drop rows with NaT or non-finite values to avoid plotting issues
            forecast_df = forecast_df.dropna(subset=['Date', 'Predicted_PTS', 'Lower_CI', 'Upper_CI'])
            forecast_df = forecast_df[(forecast_df['Predicted_PTS'].apply(np.isfinite)) &
                                      (forecast_df['Lower_CI'].apply(np.isfinite)) &
                                      (forecast_df['Upper_CI'].apply(np.isfinite))]
            
            # Final strict check to ensure no NaN or non-finite values are present
            forecast_df = forecast_df.dropna()
            forecast_df = forecast_df[(forecast_df['Predicted_PTS'].apply(np.isfinite)) &
                                      (forecast_df['Lower_CI'].apply(np.isfinite)) &
                                      (forecast_df['Upper_CI'].apply(np.isfinite))]
            
            # Plot predictions and confidence intervals
            plt.plot(forecast_df['Date'], forecast_df['Predicted_PTS'], label='Predicted Points', color='red', linestyle='--')
            plt.fill_between(forecast_df['Date'], forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='red', alpha=0.2, label='95% Confidence Interval')

            plt.title(f'Points Prediction for {team}')
            plt.xlabel('Date')
            plt.ylabel('Points')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No forecast data available for team {team}.")
    
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate a performance report for all teams."""
        report_data = []
        
        for team in self.evaluation_metrics:
            metrics = self.evaluation_metrics[team]
            report_data.append({
                'Team': team,
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'Average_Predicted_Points': self.team_forecasts[team]['Predicted_PTS'].mean()
            })
        
        return pd.DataFrame(report_data).sort_values('RMSE')

# Example usage
def main():
    predictor = NBAPointsPredictor('/Users/matthewfox/ARIMA/traditional.csv')
    predictor.load_and_prepare_data()
    
    # Train a single team for testing purposes
    predictor.train_single_team('BOS')
    
    # Visualize predictions for a specific team
    predictor.visualize_team_prediction('BOS')
    
    # Generate and display performance report
    performance_report = predictor.generate_performance_report()
    print("\nModel Performance Report:")
    print(performance_report)

if __name__ == "__main__":
    main()