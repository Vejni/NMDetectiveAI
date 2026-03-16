"""
NMDetective-A: Random Forest Regressor for NMD efficiency prediction.

This model uses a Random Forest trained on multiple genomic features as described
in the original NMDetective paper. Features are expected to be pre-computed by
the process_PTC_dataset function in data.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Optional


class NMDetectiveA:
    """
    NMDetective-A: Random Forest Regressor for NMD efficiency prediction.
    
    This model uses a Random Forest trained on multiple genomic features:
    - InLastExon: Whether PTC is in the last exon (boolean)
    - 50ntToLastEJ: Whether PTC is within 50nt of last exon junction (boolean)
    - DistanceToStart: Distance from PTC to coding start, capped at 1000 nt
    - ExonLength: Length of exon containing PTC (nt, excluding 3'UTR for last exon)
    - PTC_EJC_dist: Distance to downstream EJC (from 5' border, 50nt from junction)
    - DistanceToWTStop: Distance to wildtype stop codon (nt)
    - RNAHalfLife: RNA half-life 
    
    Note: All features should be pre-computed by process_PTC_dataset in data.py
    """
    
    def __init__(
        self, 
        n_estimators: int = 100000, 
        max_features: int = 1,
        random_state: Optional[int] = 42,
        **rf_kwargs
    ):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest (paper uses 100,000)
            max_features: Number of features to consider at each split (paper uses 1)
            random_state: Random seed for reproducibility
            **rf_kwargs: Additional arguments for RandomForestRegressor
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            **rf_kwargs
        )
        self.feature_cols_ = None
        self.is_fitted_ = False
    
    def _prepare_features(
        self, 
        df: pd.DataFrame, 
        is_training: bool = False
    ) -> pd.DataFrame:
        """
        Prepare and validate features for the model.
        
        Expects DataFrame to already have standardized column names from data.py:
        - InLastExon, 50ntToLastEJ, DistanceToStart, ExonLength
        - PTC_EJC_dist, DistanceToWTStop, RNAHalfLife
        
        Args:
            df: Input DataFrame with standardized columns
            is_training: Whether this is for training (to store feature names)
            
        Returns:
            DataFrame with processed features
        """
        df = df.copy()
        
        # Define final feature columns (removed VAF)
        feature_cols = [
            'InLastExon', '50ntToLastEJ', 'DistanceToStart', 
            'ExonLength', 'PTC_EJC_dist', 'DistanceToWTStop',
            'RNAHalfLife'
        ]
        
        # Check required columns
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. "
                           "Make sure data is processed with process_PTC_dataset first.")
        
        # Ensure boolean columns are numeric
        df['InLastExon'] = df['InLastExon'].astype(int)
        df['50ntToLastEJ'] = df['50ntToLastEJ'].astype(int)
        
        if is_training:
            self.feature_cols_ = feature_cols
        
        return df[feature_cols]
    
    def fit(self, df: pd.DataFrame, label_col: str = "NMD") -> "NMDetectiveA":
        """
        Fit the Random Forest model.
        
        Args:
            df: Training DataFrame with required features
            label_col: Name of the target column (default: "NMD")
            
        Returns:
            self
        """
        X = self._prepare_features(df, is_training=True)
        y = df[label_col].values
        
        self.model.fit(X, y)
        self.is_fitted_ = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict NMD efficiency using the Random Forest model.
        
        Args:
            df: DataFrame with required features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._prepare_features(df, is_training=False)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the fitted model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        return dict(zip(self.feature_cols_, self.model.feature_importances_))
