"""
Advanced Deep Learning Models for Institutional Time Series Forecasting
========================================================================

State-of-the-art architectures based on published research papers.
Each model implements exact specifications from the original papers
with financial-specific enhancements.

Models:
-------
- TemporalFusionTransformerV2: Google's TFT for interpretable forecasting
- NBeatsV2: N-BEATS with generic and interpretable architectures
- NHiTSV2: Hierarchical interpolation for efficient long-horizon
- InformerV2: O(L log L) attention for very long sequences
- DeepARV2: Probabilistic autoregressive forecasting

Trainer:
--------
- ProductionTrainer: Walk-forward CV, conformal prediction, checkpointing

References:
-----------
1. Lim et al., 2021 - Temporal Fusion Transformers (Google)
2. Oreshkin et al., 2020 - N-BEATS (ServiceNow/Element AI)
3. Challu et al., 2022 - N-HiTS (Nixtla)
4. Zhou et al., 2021 - Informer (AAAI 2021 Best Paper)
5. Salinas et al., 2020 - DeepAR (Amazon)
"""

# Lazy imports to avoid loading all models at once
def __getattr__(name):
    """Lazy import for heavy models."""
    
    # Aliases - map short names to V2 versions
    aliases = {
        'TemporalFusionTransformer': 'TemporalFusionTransformerV2',
        'NBeats': 'NBeatsV2', 
        'NHiTS': 'NHiTSV2',
        'Informer': 'InformerV2',
        'DeepAR': 'DeepARV2',
    }
    if name in aliases:
        name = aliases[name]
    
    # TFT components
    if name in ('TemporalFusionTransformerV2', 'TFTConfig', 'TFTLoss', 
                'quantile_loss', 'GatedResidualNetwork', 'VariableSelectionNetwork',
                'GatedLinearUnit', 'StaticCovariateEncoders', 
                'InterpretableMultiHeadAttention', 'TemporalSelfAttention', 'QuantileOutput'):
        from .temporal_fusion_transformer import (
            TemporalFusionTransformerV2, TFTConfig, TFTLoss,
            quantile_loss, GatedResidualNetwork, VariableSelectionNetwork,
            GatedLinearUnit, StaticCovariateEncoders,
            InterpretableMultiHeadAttention, TemporalSelfAttention, QuantileOutput
        )
        return locals()[name]
    
    # N-BEATS components
    if name in ('NBeatsV2', 'NBeatsConfig', 'NBeatsGeneric', 
                'NBeatsInterpretable', 'NBeatsEnsemble', 'NBeatsLoss',
                'GenericBlock', 'TrendBlock', 'SeasonalityBlock', 'NBEATSBlock'):
        from .nbeats import (
            NBeatsV2, NBeatsConfig, NBeatsGeneric,
            NBeatsInterpretable, NBeatsEnsemble, NBeatsLoss,
            GenericBlock, TrendBlock, SeasonalityBlock, NBEATSBlock
        )
        return locals()[name]
    
    # N-HiTS components
    if name in ('NHiTSV2', 'NHiTSConfig', 'NHiTSProbabilistic',
                'NHiTSWithExogenous', 'NHiTSLoss', 'NHiTSBlock'):
        from .nhits import (
            NHiTSV2, NHiTSConfig, NHiTSProbabilistic,
            NHiTSWithExogenous, NHiTSLoss, NHiTSBlock
        )
        return locals()[name]
    
    # Informer components  
    if name in ('InformerV2', 'InformerConfig', 'InformerLoss', 'ProbSparseAttention'):
        from .informer import (
            InformerV2, InformerConfig, InformerLoss, ProbSparseAttention
        )
        return locals()[name]
    
    # DeepAR components
    if name in ('DeepARV2', 'DeepARConfig', 'DeepARLoss',
                'NormalOutput', 'StudentTOutput'):
        from .deepar import (
            DeepARV2, DeepARConfig, DeepARLoss,
            NormalOutput, StudentTOutput
        )
        return locals()[name]
    
    # Trainer components
    if name in ('ProductionTrainer', 'TrainerConfig', 'ValidationStrategy',
                'ConformalPredictor', 'AdaptiveConformalPredictor',
                'WalkForwardValidator', 'MetricsTracker'):
        from .production_trainer import (
            ProductionTrainer, TrainerConfig, ValidationStrategy,
            ConformalPredictor, AdaptiveConformalPredictor,
            WalkForwardValidator, MetricsTracker
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # TFT
    'TemporalFusionTransformerV2',
    'TFTConfig',
    'TFTLoss',
    'quantile_loss',
    'GatedLinearUnit',
    'GatedResidualNetwork',
    'VariableSelectionNetwork',
    'StaticCovariateEncoders',
    'InterpretableMultiHeadAttention',
    'TemporalSelfAttention',
    'QuantileOutput',
    
    # N-BEATS
    'NBeatsV2',
    'NBeatsConfig',
    'NBeatsGeneric',
    'NBeatsInterpretable',
    'NBeatsEnsemble',
    'NBeatsLoss',
    'NBEATSBlock',
    'GenericBlock',
    'TrendBlock',
    'SeasonalityBlock',
    
    # N-HiTS
    'NHiTSV2',
    'NHiTSConfig',
    'NHiTSProbabilistic',
    'NHiTSWithExogenous',
    'NHiTSLoss',
    'NHiTSBlock',
    
    # Informer
    'InformerV2',
    'InformerConfig',
    'InformerLoss',
    'ProbSparseAttention',
    
    # DeepAR
    'DeepARV2',
    'DeepARConfig',
    'DeepARLoss',
    'NormalOutput',
    'StudentTOutput',
    
    # Trainer
    'ProductionTrainer',
    'TrainerConfig',
    'ValidationStrategy',
    'ConformalPredictor',
    'AdaptiveConformalPredictor',
    'WalkForwardValidator',
    'MetricsTracker',
    
    # Aliases (shorter names)
    'TemporalFusionTransformer',
    'NBeats',
    'NHiTS',
    'Informer',
    'DeepAR',
]

# ===== Aliases for simpler imports =====
# These map to the V2 versions
def _create_alias(name):
    """Create aliases mapping short names to V2 versions."""
    aliases = {
        'TemporalFusionTransformer': 'TemporalFusionTransformerV2',
        'NBeats': 'NBeatsV2', 
        'NHiTS': 'NHiTSV2',
        'Informer': 'InformerV2',
        'DeepAR': 'DeepARV2',
    }
    return aliases.get(name, name)
