"""
NMDetective-B: Simple decision tree predictor for NMD efficiency.

This model uses a fixed decision tree structure based on key genomic features
as described in the original NMDetective paper.
"""

import numpy as np
import pandas as pd


class NMDetectiveB:
    """
    NMDetective-B: Simple decision tree predictor based on fixed rules.
    
    This model uses a fixed decision tree structure based on key genomic features:
    - InLastExon: Whether PTC is in the last exon
    - DistanceToStart: Distance from PTC to coding start (nt)
    - ExonLength: Length of exon containing PTC (nt)
    - 50ntToLastEJ: Whether PTC is within 50nt of last exon junction
    """
    
    def __init__(self):
        self.means_ = None
        self.is_fitted_ = False
    
    def fit(self, df: pd.DataFrame, label_col: str = "NMD") -> "NMDetectiveB":
        """
        Fit the decision tree by computing mean values for each leaf node.
        
        Args:
            df: Training DataFrame with required features
            label_col: Name of the target column (default: "NMD")
            
        Returns:
            self
        """
        required_cols = ["InLastExon", "DistanceToStart", "ExonLength", "50ntToLastEJ"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Compute means for each decision tree leaf
        self.means_ = {
            'InLastExon_True': df[df['InLastExon'] == True][label_col].mean(),
            'InLastExon_False_DistanceStart_150': df[
                (df['InLastExon'] == False) & 
                (df['DistanceToStart'] < 150)
            ][label_col].mean(),
            'InLastExon_False_DistanceStart_gt150_ExonLength_gt407': df[
                (df['InLastExon'] == False) & 
                (df['DistanceToStart'] > 150) & 
                (df['ExonLength'] > 407)
            ][label_col].mean(),
            'InLastExon_False_DistanceStart_gt150_ExonLength_lt407_50nt_False': df[
                (df['InLastExon'] == False) & 
                (df['DistanceToStart'] > 150) & 
                (df['ExonLength'] < 407) & 
                (df['50ntToLastEJ'] == False)
            ][label_col].mean(),
            'InLastExon_False_DistanceStart_gt150_ExonLength_lt407_50nt_True': df[
                (df['InLastExon'] == False) & 
                (df['DistanceToStart'] > 150) & 
                (df['ExonLength'] < 407) & 
                (df['50ntToLastEJ'] == True)
            ][label_col].mean()
        }
        
        self.is_fitted_ = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict NMD efficiency using the decision tree rules.
        
        Args:
            df: DataFrame with required features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
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
