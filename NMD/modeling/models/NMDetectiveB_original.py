"""
NMDetective-C: Fixed decision tree predictor for NMD efficiency.

This model uses the same decision tree structure as NMDetective-B but with
fixed leaf values instead of computing them from training data.
"""

import numpy as np
import pandas as pd


class NMDetectiveB_original:
    """
    NMDetective-B_original: Fixed decision tree predictor with pre-defined leaf values.
    
    This model uses the same decision tree structure as NMDetective-B:
    - InLastExon: Whether PTC is in the last exon
    - DistanceToStart: Distance from PTC to coding start (nt)
    - ExonLength: Length of exon containing PTC (nt)
    - 50ntToLastEJ: Whether PTC is within 50nt of last exon junction
    
    But uses fixed leaf values determined a priori.
    """
    
    def __init__(self):
        # Fixed leaf values for each decision tree outcome
        self.means_ = {
            'InLastExon_True': 0.0,
            'InLastExon_False_DistanceStart_150': 0.12,
            'InLastExon_False_DistanceStart_gt150_ExonLength_gt407': 0.41,
            'InLastExon_False_DistanceStart_gt150_ExonLength_lt407_50nt_False': 0.65,
            'InLastExon_False_DistanceStart_gt150_ExonLength_lt407_50nt_True': 0.20,
        }
        self.is_fitted_ = True  # Always fitted since values are fixed
    
    def fit(self, df: pd.DataFrame, label_col: str = "NMD") -> "NMDetectiveB_original":
        """
        Fit method (no-op since values are fixed).
        
        Args:
            df: Training DataFrame (not used)
            label_col: Name of the target column (not used)
            
        Returns:
            self
        """
        # No fitting needed - values are fixed
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict NMD efficiency using the fixed decision tree rules.
        
        Args:
            df: DataFrame with required features
            
        Returns:
            Array of predictions
        """
        required_cols = ["InLastExon", "DistanceToStart", "ExonLength", "50ntToLastEJ"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        predictions = df.apply(self._predict_single, axis=1).values
        return predictions
    
    def _predict_single(self, row: pd.Series) -> float:
        """Apply decision tree rules to a single row."""
        if row['InLastExon']:
            return self.means_['InLastExon_True']
        elif row['DistanceToStart'] < 150:
            return self.means_['InLastExon_False_DistanceStart_150']
        elif row['ExonLength'] > 407:
            return self.means_['InLastExon_False_DistanceStart_gt150_ExonLength_gt407']
        else:
            if row['50ntToLastEJ']:
                return self.means_['InLastExon_False_DistanceStart_gt150_ExonLength_lt407_50nt_True']
            else:
                return self.means_['InLastExon_False_DistanceStart_gt150_ExonLength_lt407_50nt_False']
