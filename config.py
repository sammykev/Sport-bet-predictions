"""Configuration management for the prediction system."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
PREDICTIONS_DIR = BASE_DIR / 'predictions'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'epl': {
        'history_file': str(DATA_DIR / 'epl-final.csv'),
        'fixtures_file': str(DATA_DIR / 'epl-2025.csv'),
        'output_file': str(PREDICTIONS_DIR / 'epl_predictions.csv'),
    },
    'ucl': {
        'history_pattern': str(DATA_DIR / 'champions-league-*.csv'),
        'output_file': str(PREDICTIONS_DIR / 'cl_predictions.csv'),
    }
}

# Model configuration
MODEL_CONFIG = {
    'n_estimators': 100,
    'min_samples_leaf': 5,
    'random_state': 42,
    'save_dir': str(MODELS_DIR),
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'rolling_window': 5,  # Last 5 games for form/goals average
    'default_goals_avg': 1.5,
    'default_form': 1.0,
    'default_form_ucl': 1.3,  # Champions League default (games are less frequent)
    'predictors': ['HomeCode', 'AwayCode', 'Home_G_Avg', 'Away_G_Avg', 'Home_Form', 'Away_Form'],
}

# Prediction confidence thresholds
PREDICTION_CONFIG = {
    'win_threshold': 0.55,
    'draw_threshold': 0.33,
    'combined_threshold': 0.75,
    'banker_threshold': 0.80,  # Over 1.5 goals
    'over25_threshold': 0.55,  # Over 2.5 goals
    'over15_threshold': 0.65,  # Over 1.5 goals (Asian)
    'under25_threshold': 0.60,  # Under 2.5 goals
}

# Output configuration
OUTPUT_CONFIG = {
    'epl_top_matches': 10,  # Top 10 matches for EPL
    'ucl_top_matches': 18,  # Top 18 matches for UCL
    'date_format': '%d-%m-%Y',  # For EPL output
    'date_format_ucl': '%Y-%m-%d',  # For UCL output
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(LOGS_DIR / 'predictions.log'),
}
