from dataclasses import dataclass, field
from typing import Literal, List
from pathlib import Path


@dataclass
class DatasetConfig:
    """
    Configuration for PTC dataset preprocessing.
    """
    
    # ===== Somatic Overlap Removal =====
    somatic_overlap_removal: Literal["remove_from_somatic", "remove_from_germline", "none"] = "remove_from_germline"
    
    # ===== Expression & CV Filtering (somatic only) =====
    apply_expression_filter: bool = True
    min_tpm_train: float = 1.0  # Min TPM for training chromosomes
    max_cv_train: float = 1  # Max coefficient of variation for training chromosomes
    min_tpm_val: float = 5.0  # Min TPM for validation chromosomes
    max_cv_val: float = 0.5  # Max coefficient of variation for validation chromosomes
    
    # ===== Splice Site Filtering =====
    apply_splice_filter: bool = True
    splice_proximity_threshold: int = 3  # nt from splice site boundaries
    
    # ===== VAF Filtering (germline only) =====
    apply_vaf_filter: bool = True
    min_vaf: float = 0.0
    max_vaf: float = 0.001
    
    # ===== Frameshift Correction =====
    apply_frameshift_correction: bool = True
    
    # ===== Regression Correction =====
    apply_regression_correction: bool = True
    regression_separate_by_variant_type: bool = False  # If True, fit separate models for SNVs and indels
    regression_predictors: List[str] = field(default_factory=lambda: [
        "tissue_PCs",  # Will use tissue_PC1, tissue_PC2, tissue_PC3, tissue_PC4
        "coeff_var", "median_TPM_exp_transcript"
    ])  # Options: "tissue_PCs", "RNAHalfLife", "VAF", "LOUEF_score", "coeff_var", "median_TPM_exp_transcript"

    # ===== Data Centering =====
    center_nmd_efficiency: bool = True
    
    # ===== Variant Aggregation =====
    aggregation_method: Literal["median", "mean"] = "median"  # std, q25, q75 always computed
    
    # ===== Threshold Filtering =====
    apply_threshold_filter: bool = True
    nmd_efficiency_threshold: float = 4.0
    
    # ===== Normalization =====
    normalize_nmd_efficiency: bool = True
    
    # ===== Transcript Length Filtering =====
    apply_transcript_length_filter: bool = True
    max_transcript_length: int = 20000
    apply_length_filter_to_val_chrs: bool = False  # If True, applies to val chrs; if False, exempts val chrs
    apply_length_filter_to_germline: bool = False  # If True, applies same logic as somatic
    
    def save_to_file(self, output_path: Path):
        """Save configuration parameters to a text file in training_parameters.txt format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=== PTC Dataset Preprocessing Parameters ===\n\n")
            
            f.write("=== Somatic Overlap Removal ===\n")
            f.write(f"somatic_overlap_removal: {self.somatic_overlap_removal}\n\n")
            
            f.write("=== Expression & CV Filtering (somatic only) ===\n")
            f.write(f"apply_expression_filter: {self.apply_expression_filter}\n")
            f.write(f"min_tpm_train: {self.min_tpm_train}\n")
            f.write(f"max_cv_train: {self.max_cv_train}\n")
            f.write(f"min_tpm_val: {self.min_tpm_val}\n")
            f.write(f"max_cv_val: {self.max_cv_val}\n\n")
            
            f.write("=== Splice Site Filtering ===\n")
            f.write(f"apply_splice_filter: {self.apply_splice_filter}\n")
            f.write(f"splice_proximity_threshold: {self.splice_proximity_threshold}\n\n")
            
            f.write("=== VAF Filtering (germline only) ===\n")
            f.write(f"apply_vaf_filter: {self.apply_vaf_filter}\n")
            f.write(f"min_vaf: {self.min_vaf}\n")
            f.write(f"max_vaf: {self.max_vaf}\n\n")
            
            f.write("=== Frameshift Correction ===\n")
            f.write(f"apply_frameshift_correction: {self.apply_frameshift_correction}\n\n")
            
            f.write("=== Regression Correction ===\n")
            f.write(f"apply_regression_correction: {self.apply_regression_correction}\n")
            f.write(f"regression_separate_by_variant_type: {self.regression_separate_by_variant_type}\n")
            f.write(f"regression_predictors: {self.regression_predictors}\n\n")
            
            f.write("=== Data Centering ===\n")
            f.write(f"center_nmd_efficiency: {self.center_nmd_efficiency}\n\n")
            
            f.write("=== Variant Aggregation ===\n")
            f.write(f"aggregation_method: {self.aggregation_method}\n\n")
            
            f.write("=== Threshold Filtering ===\n")
            f.write(f"apply_threshold_filter: {self.apply_threshold_filter}\n")
            f.write(f"nmd_efficiency_threshold: {self.nmd_efficiency_threshold}\n\n")
            
            f.write("=== Normalization ===\n")
            f.write(f"normalize_nmd_efficiency: {self.normalize_nmd_efficiency}\n\n")
            
            f.write("=== Transcript Length Filtering ===\n")
            f.write(f"apply_transcript_length_filter: {self.apply_transcript_length_filter}\n")
            f.write(f"max_transcript_length: {self.max_transcript_length}\n")
            f.write(f"apply_length_filter_to_val_chrs: {self.apply_length_filter_to_val_chrs}\n")
            f.write(f"apply_length_filter_to_germline: {self.apply_length_filter_to_germline}\n")
