# Training Infrastructure
"""
Production-ready training infrastructure for NUBLE.

Key components:
- RealDataTrainer: Train on real market data from Polygon
- WalkForwardTrainer: Rolling window retraining
- ModelTrainer: Legacy model trainer (from training.py)

Usage:
    from institutional.ml.training import RealDataTrainer, train_on_real_data
    
    trainer = RealDataTrainer(api_key=os.environ['POLYGON_API_KEY'])
    result = await trainer.train_and_validate('SPY')
"""

from .real_data_trainer import (
    RealDataTrainer,
    train_on_real_data,
    TrainingResult,
)

from .walk_forward import (
    WalkForwardTrainer,
    WalkForwardResult,
)

# Import legacy training classes from the old training.py (renamed)
from .legacy import (
    ModelTrainer,
    CrossValidator,
    HyperparameterOptimizer,
    WalkForwardValidator,
    TrainingConfig,
)

__all__ = [
    # New production-ready classes
    'RealDataTrainer',
    'train_on_real_data',
    'TrainingResult',
    'WalkForwardTrainer',
    'WalkForwardResult',
    # Legacy classes (for backward compatibility)
    'ModelTrainer',
    'CrossValidator',
    'HyperparameterOptimizer',
    'WalkForwardValidator',
    'TrainingConfig',
]
