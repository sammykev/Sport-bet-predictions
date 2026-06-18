"""Model training and prediction utilities."""
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.logger import setup_logger
from config import MODEL_CONFIG, FEATURE_CONFIG

logger = setup_logger(__name__)


class ModelManager:
    """Manages model training and prediction."""
    
    def __init__(self, league='epl', config=None):
        self.league = league
        self.config = config or MODEL_CONFIG
        self.models = {}
        self.model_dir = Path(self.config['save_dir'])
        self.model_dir.mkdir(exist_ok=True)
    
    def train(self, train_data, league_type='epl'):
        """Train Random Forest models for win, over 1.5, and over 2.5 predictions."""
        logger.info(f"Training {league_type.upper()} models...")
        
        predictors = FEATURE_CONFIG['predictors']
        
        try:
            # Win model
            self.models['win'] = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state']
            )
            self.models['win'].fit(train_data[predictors], train_data['Target_Win'])
            logger.info("Trained win model")
            
            # Over 1.5 goals model
            self.models['over15'] = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state']
            )
            self.models['over15'].fit(train_data[predictors], train_data['Target_Over15'])
            logger.info("Trained over 1.5 model")
            
            # Over 2.5 goals model
            self.models['over25'] = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state']
            )
            self.models['over25'].fit(train_data[predictors], train_data['Target_Over25'])
            logger.info("Trained over 2.5 model")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, test_data):
        """Generate predictions for test data."""
        predictors = FEATURE_CONFIG['predictors']
        
        try:
            probs_win = self.models['win'].predict_proba(test_data[predictors])
            probs_over15 = self.models['over15'].predict_proba(test_data[predictors])[:, 1]
            probs_over25 = self.models['over25'].predict_proba(test_data[predictors])[:, 1]
            
            return probs_win, probs_over15, probs_over25
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def save_models(self, suffix=''):
        """Save trained models to disk."""
        try:
            for model_name, model in self.models.items():
                path = self.model_dir / f"{self.league}_{model_name}{suffix}.pkl"
                joblib.dump(model, path)
                logger.info(f"Saved model: {path}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def load_models(self, suffix=''):
        """Load trained models from disk."""
        try:
            for model_name in ['win', 'over15', 'over25']:
                path = self.model_dir / f"{self.league}_{model_name}{suffix}.pkl"
                if path.exists():
                    self.models[model_name] = joblib.load(path)
                    logger.info(f"Loaded model: {path}")
                else:
                    logger.warning(f"Model not found: {path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
