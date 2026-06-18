"""Feature engineering utilities for match prediction."""
import pandas as pd
import numpy as np
from src.logger import setup_logger
from config import FEATURE_CONFIG

logger = setup_logger(__name__)


class FeatureEngineer:
    """Handles feature engineering for prediction models."""
    
    def __init__(self, config=None):
        self.config = config or FEATURE_CONFIG
    
    @staticmethod
    def create_targets(data):
        """Create target variables from match results."""
        data['TotalGoals'] = data['FTHG'] + data['FTAG']
        data['Target_Over15'] = (data['TotalGoals'] >= 2).astype(int)
        data['Target_Over25'] = (data['TotalGoals'] >= 3).astype(int)
        
        # Map results to numeric targets
        if 'Result' in data.columns:
            data['Target_Win'] = data['Result'].map({'H': 2, 'D': 1, 'A': 0})
        elif 'Outcome' in data.columns:
            data['Target_Win'] = data['Outcome'].map({'H': 2, 'D': 1, 'A': 0})
        
        return data
    
    @staticmethod
    def encode_teams(data):
        """Create numeric encodings for teams."""
        all_teams = pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()
        encoder = {team: i for i, team in enumerate(all_teams)}
        
        data['HomeCode'] = data['HomeTeam'].map(encoder)
        data['AwayCode'] = data['AwayTeam'].map(encoder)
        
        return data, encoder
    
    def engineer_epl_features(self, history, played, upcoming):
        """Engineer features for EPL predictions."""
        logger.info("Engineering EPL features...")
        
        # Prepare training data
        h_sub = history[['MatchDate', 'HomeTeam', 'AwayTeam', 'FullTimeResult',
                         'FullTimeHomeGoals', 'FullTimeAwayGoals']].copy()
        h_sub.columns = ['Date', 'HomeTeam', 'AwayTeam', 'Result', 'FTHG', 'FTAG']
        
        p_sub = played[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'FTHG', 'FTAG']].copy()
        p_sub.columns = ['Date', 'HomeTeam', 'AwayTeam', 'Result', 'FTHG', 'FTAG']
        
        full_data = pd.concat([h_sub, p_sub], ignore_index=True)
        full_data.sort_values('Date', inplace=True)
        
        # Create targets
        full_data = self.create_targets(full_data)
        
        # Encode teams
        full_data, encoder = self.encode_teams(full_data)
        
        # Rolling stats
        full_data = self._add_rolling_stats(full_data, result_col='Result')
        
        return full_data, encoder
    
    def engineer_ucl_features(self, played, upcoming):
        """Engineer features for UCL predictions."""
        logger.info("Engineering UCL features...")
        
        # Combine for feature engineering
        all_rows = pd.concat([played, upcoming], ignore_index=True)
        all_rows.sort_values('Date', inplace=True)
        
        # Create targets
        all_rows = self.create_targets(all_rows)
        
        # Encode teams
        all_rows, encoder = self.encode_teams(all_rows)
        
        # Rolling stats
        all_rows = self._add_rolling_stats(all_rows, result_col='Outcome', is_ucl=True)
        
        # Re-split
        train = all_rows[all_rows['Outcome'] != 'Unplayed'].copy()
        predict = all_rows[all_rows['Outcome'] == 'Unplayed'].copy()
        
        return train, predict, encoder
    
    def _add_rolling_stats(self, data, result_col='Result', is_ucl=False):
        """Add rolling statistics (form and goals average)."""
        all_teams = pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()
        team_stats = {team: {'goals': [], 'points': []} for team in all_teams}
        
        h_g_avg, a_g_avg, h_form, a_form = [], [], [], []
        
        default_goals = self.config['default_goals_avg']
        default_form = self.config['default_form_ucl'] if is_ucl else self.config['default_form']
        window = self.config['rolling_window']
        
        for idx, row in data.iterrows():
            ht, at = row['HomeTeam'], row['AwayTeam']
            
            # Get past stats
            h_past_g = team_stats[ht]['goals'][-window:]
            a_past_g = team_stats[at]['goals'][-window:]
            h_past_p = team_stats[ht]['points'][-window:]
            a_past_p = team_stats[at]['points'][-window:]
            
            # Calculate averages
            h_g_avg.append(sum(h_past_g) / len(h_past_g) if h_past_g else default_goals)
            a_g_avg.append(sum(a_past_g) / len(a_past_g) if a_past_g else default_goals)
            h_form.append(sum(h_past_p) / len(h_past_p) if h_past_p else default_form)
            a_form.append(sum(a_past_p) / len(a_past_p) if a_past_p else default_form)
            
            # Update stats with current result
            outcome = row[result_col]
            if outcome != 'Unplayed':
                if outcome == 'H':
                    team_stats[ht]['points'].append(3)
                    team_stats[at]['points'].append(0)
                elif outcome == 'A':
                    team_stats[ht]['points'].append(0)
                    team_stats[at]['points'].append(3)
                else:  # Draw
                    team_stats[ht]['points'].append(1)
                    team_stats[at]['points'].append(1)
                
                team_stats[ht]['goals'].append(row['FTHG'])
                team_stats[at]['goals'].append(row['FTAG'])
        
        data['Home_G_Avg'] = h_g_avg
        data['Away_G_Avg'] = a_g_avg
        data['Home_Form'] = h_form
        data['Away_Form'] = a_form
        
        return data
