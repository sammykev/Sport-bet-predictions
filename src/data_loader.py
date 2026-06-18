"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Base class for loading and cleaning match data."""
    
    def __init__(self, config):
        self.config = config
    
    @staticmethod
    def parse_score(score_str):
        """Parse score string in format 'X - Y' to (X, Y)."""
        if pd.isna(score_str):
            return np.nan, np.nan
        try:
            parts = str(score_str).split('-')
            return int(parts[0].strip()), int(parts[1].strip())
        except Exception as e:
            logger.warning(f"Failed to parse score '{score_str}': {e}")
            return np.nan, np.nan
    
    @staticmethod
    def determine_outcome(fthg, ftag):
        """Determine match outcome (H/D/A) from goals."""
        if pd.isna(fthg) or pd.isna(ftag):
            return 'Unplayed'
        if fthg > ftag:
            return 'H'
        elif fthg < ftag:
            return 'A'
        else:
            return 'D'


class EPLDataLoader(DataLoader):
    """EPL-specific data loader."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name_map = {
            'Man Utd': 'Man United',
            'Spurs': 'Tottenham',
            'West Ham': 'West Ham',
            'Newcastle': 'Newcastle',
            'Wolves': 'Wolves',
            "Nott'm Forest": "Nott'm Forest",
            'Leeds': 'Leeds',
            'Sunderland': 'Sunderland',
            'Burnley': 'Burnley',
            'Brighton': 'Brighton',
            'Leicester': 'Leicester',
            'Man City': 'Man City'
        }
    
    def load_data(self):
        """Load and process EPL data."""
        logger.info("Loading EPL data...")
        
        try:
            # Load history
            history = pd.read_csv(self.config['history_file'])
            history['MatchDate'] = pd.to_datetime(history['MatchDate'])
            logger.info(f"Loaded {len(history)} historical matches")
            
            # Load fixtures
            fixtures = pd.read_csv(self.config['fixtures_file'])
            fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
            logger.info(f"Loaded {len(fixtures)} fixtures")
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        
        # Parse results
        scores = fixtures['Result'].apply(self.parse_score)
        fixtures['FTHG'] = [x[0] for x in scores]
        fixtures['FTAG'] = [x[1] for x in scores]
        fixtures['FullTimeResult'] = fixtures.apply(
            lambda row: self.determine_outcome(row['FTHG'], row['FTAG']), axis=1
        )
        
        # Clean team names
        fixtures['HomeTeam'] = fixtures['Home Team'].apply(lambda x: self.name_map.get(x, x))
        fixtures['AwayTeam'] = fixtures['Away Team'].apply(lambda x: self.name_map.get(x, x))
        
        # Split into played and upcoming
        played = fixtures[fixtures['FullTimeResult'] != 'Unplayed'].copy()
        upcoming = fixtures[fixtures['FullTimeResult'] == 'Unplayed'].copy()
        
        logger.info(f"Split fixtures: {len(played)} played, {len(upcoming)} upcoming")
        
        return history, played, upcoming


class UCLDataLoader(DataLoader):
    """Champions League-specific data loader."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name_map = {
            'Man. City': 'Man City', 'Manchester City': 'Man City',
            'Man. United': 'Man United', 'Manchester United': 'Man United',
            'Bayern München': 'Bayern Munich', 'Bayern': 'Bayern Munich',
            'B. Dortmund': 'Dortmund', 'Borussia Dortmund': 'Dortmund',
            'Atlético': 'Atlético Madrid', 'Atleti': 'Atlético Madrid',
            'Atlético de Madrid': 'Atlético Madrid',
            'Paris Saint-Germain': 'Paris',
            'Internazionale': 'Inter',
            'Tottenham Hotspur': 'Tottenham',
            'RB Leipzig': 'Leipzig',
            'FC Porto': 'Porto',
            'CSKA Moskva': 'CSKA Moscow',
            'Shakhtar Donetsk': 'Shakhtar',
            'LOSC': 'Lille',
            'Crvena zvezda': 'Crvena Zvezda',
            'GNK Dinamo': 'Dinamo Zagreb'
        }
    
    def load_data(self):
        """Load and process UCL data from multiple files."""
        logger.info("Loading Champions League data...")
        
        try:
            files = sorted(glob.glob(self.config['history_pattern']))
            if not files:
                raise FileNotFoundError(f"No files matching pattern: {self.config['history_pattern']}")
            
            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    df.columns = [c.strip() for c in df.columns]
                    dfs.append(df)
                    logger.info(f"Loaded {f}: {len(df)} matches")
                except Exception as e:
                    logger.warning(f"Skipping {f}: {e}")
            
            if not dfs:
                raise ValueError("No valid data files found")
            
            full_data = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to load UCL data: {e}")
            raise
        
        # Parse dates and results
        full_data['Date'] = pd.to_datetime(full_data['Date'], dayfirst=True, errors='coerce')
        full_data.sort_values('Date', inplace=True)
        
        scores = full_data['Result'].apply(self.parse_score)
        full_data['FTHG'] = [x[0] for x in scores]
        full_data['FTAG'] = [x[1] for x in scores]
        full_data['Outcome'] = full_data.apply(
            lambda row: self.determine_outcome(row['FTHG'], row['FTAG']), axis=1
        )
        
        # Clean team names
        full_data['HomeTeam'] = full_data['Home Team'].apply(lambda x: self.name_map.get(str(x).strip(), str(x).strip()))
        full_data['AwayTeam'] = full_data['Away Team'].apply(lambda x: self.name_map.get(str(x).strip(), str(x).strip()))
        
        # Split
        played = full_data[full_data['Outcome'] != 'Unplayed'].copy()
        upcoming = full_data[full_data['Outcome'] == 'Unplayed'].copy()
        
        logger.info(f"Split data: {len(played)} played, {len(upcoming)} upcoming")
        
        return played, upcoming
