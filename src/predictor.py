"""Core prediction logic and betting recommendations."""
import pandas as pd
from src.logger import setup_logger
from config import PREDICTION_CONFIG

logger = setup_logger(__name__)


class BettingPredictor:
    """Generates betting predictions based on model probabilities."""
    
    def __init__(self, config=None):
        self.config = config or PREDICTION_CONFIG
    
    def predict_1x2(self, p_h, p_d, p_a):
        """Predict 1X2 outcome (Win/Draw/Away)."""
        if p_h > self.config['win_threshold']:
            return "Home Win (1)"
        elif p_a > self.config['win_threshold']:
            return "Away Win (2)"
        elif p_d > self.config['draw_threshold']:
            return "Draw (X)"
        elif (p_h + p_d) > self.config['combined_threshold']:
            return "Home or Draw (1X)"
        elif (p_a + p_d) > self.config['combined_threshold']:
            return "Away or Draw (2X)"
        elif (p_h + p_a) > self.config['combined_threshold']:
            return "Home or Away (12)"
        else:
            return "Skip"
    
    def predict_goals(self, p_o15, p_o25):
        """Predict goals outcome with safe mode logic."""
        p_u25 = 1.0 - p_o25
        
        # 1. The "Banker" (Extremely Safe)
        if p_o15 > self.config['banker_threshold']:
            return "Over 1.5 Goals (Banker)"
        
        # 2. The "Refundable" 2.0
        elif p_o25 > self.config['over25_threshold']:
            return "Over 2.0 (Asian)"
        
        # 3. The "Refundable" 1.0
        elif p_o15 > self.config['over15_threshold']:
            return "Over 1.0 (Asian)"
        
        # 4. Safe Unders
        elif p_u25 > self.config['under25_threshold']:
            return "Under 3.0 (Asian)"
        
        else:
            return "Skip"
    
    def generate_predictions(self, matches_data, probs_win, probs_over15, probs_over25,
                            date_format='%d-%m-%Y'):
        """Generate complete predictions for all matches."""
        logger.info(f"Generating predictions for {len(matches_data)} matches...")
        
        results = []
        for i, (idx, row) in enumerate(matches_data.iterrows()):
            match = f"{row['HomeTeam']} vs {row['AwayTeam']}"
            date = row['Date'].strftime(date_format)
            
            # Extract probabilities
            p_h = probs_win[i][2]
            p_d = probs_win[i][1]
            p_a = probs_win[i][0]
            p_o15 = probs_over15[i]
            p_o25 = probs_over25[i]
            
            # Generate predictions
            tip_1x2 = self.predict_1x2(p_h, p_d, p_a)
            tip_goals = self.predict_goals(p_o15, p_o25)
            
            results.append({
                'Date': date,
                'Match': match,
                '1X2 Prediction': tip_1x2,
                'Goal Prediction': tip_goals,
                'Home Win Confidence': f"{p_h:.0%}",
                'Draw Confidence': f"{p_d:.0%}",
                'Away Win Confidence': f"{p_a:.0%}",
                'Goal Confidence': f"{max(p_o25, 1.0 - p_o25):.0%}",
            })
        
        return pd.DataFrame(results)
