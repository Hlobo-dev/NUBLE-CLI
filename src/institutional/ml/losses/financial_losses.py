"""
Loss functions that actually matter for trading.
MSE optimizes for prediction accuracy.
These optimize for MONEY.

Critical insight: In trading, being wrong about DIRECTION is catastrophic.
Being wrong about magnitude is acceptable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SharpeRatioLoss(nn.Module):
    """
    Directly optimize Sharpe ratio.
    
    This is what you actually care about, not MSE.
    
    Note: Sharpe is not convex, so this can be tricky to optimize.
    Use with gradient clipping and careful learning rate.
    """
    
    def __init__(self, annualization_factor: float = 252.0):
        super().__init__()
        self.annualization = annualization_factor ** 0.5
    
    def forward(self, 
                predictions: torch.Tensor, 
                returns: torch.Tensor,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [batch, horizon] or [batch]
            returns: Actual returns [batch, horizon] or [batch]
            positions: Optional pre-computed positions. If None, uses tanh(predictions)
        
        Returns:
            Negative Sharpe ratio (for minimization)
        """
        if positions is None:
            # Convert predictions to positions via tanh
            positions = torch.tanh(predictions)
        
        # Portfolio returns
        portfolio_returns = positions * returns
        
        # Flatten if multi-horizon
        portfolio_returns = portfolio_returns.flatten()
        
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() + 1e-8
        
        sharpe = mean_return / std_return * self.annualization
        
        # Return negative because we minimize
        return -sharpe


class SortinoRatioLoss(nn.Module):
    """
    Optimize Sortino ratio - only penalizes downside volatility.
    
    Better than Sharpe when you care about avoiding losses
    more than capturing gains (most real traders).
    """
    
    def __init__(self, annualization_factor: float = 252.0, mar: float = 0.0):
        super().__init__()
        self.annualization = annualization_factor ** 0.5
        self.mar = mar  # Minimum acceptable return
    
    def forward(self, 
                predictions: torch.Tensor, 
                returns: torch.Tensor) -> torch.Tensor:
        
        positions = torch.tanh(predictions)
        portfolio_returns = (positions * returns).flatten()
        
        excess_returns = portfolio_returns - self.mar
        mean_return = excess_returns.mean()
        
        # Downside deviation
        downside_returns = torch.clamp(excess_returns, max=0)
        downside_std = (downside_returns ** 2).mean().sqrt() + 1e-8
        
        sortino = mean_return / downside_std * self.annualization
        
        return -sortino


class DirectionalLoss(nn.Module):
    """
    Heavily penalize wrong direction predictions.
    
    In trading, being wrong about direction is much worse than
    being wrong about magnitude. A stock going up 1% vs 2% doesn't
    matter if you're long. A stock going down 1% when you're long
    is catastrophic.
    """
    
    def __init__(self, wrong_direction_penalty: float = 3.0):
        super().__init__()
        self.penalty = wrong_direction_penalty
    
    def forward(self, 
                predictions: torch.Tensor, 
                returns: torch.Tensor) -> torch.Tensor:
        
        # Standard MSE
        mse = (predictions - returns) ** 2
        
        # Direction check
        pred_sign = torch.sign(predictions)
        actual_sign = torch.sign(returns)
        wrong_direction = pred_sign != actual_sign
        
        # Apply penalty for wrong direction
        weights = torch.where(wrong_direction, 
                             torch.tensor(self.penalty, device=predictions.device), 
                             torch.tensor(1.0, device=predictions.device))
        
        return (weights * mse).mean()


class AsymmetricLoss(nn.Module):
    """
    Quantile loss - penalize under/over prediction asymmetrically.
    
    Use alpha > 0.5 if you care more about not missing upside.
    Use alpha < 0.5 if you care more about not overpredicting.
    """
    
    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, 
                predictions: torch.Tensor, 
                returns: torch.Tensor) -> torch.Tensor:
        
        errors = returns - predictions
        
        loss = torch.where(
            errors >= 0,
            self.alpha * errors,
            (self.alpha - 1) * errors
        )
        
        return loss.abs().mean()


class MaxDrawdownLoss(nn.Module):
    """
    Penalize strategies that have large drawdowns.
    
    Combines return maximization with drawdown penalty.
    """
    
    def __init__(self, drawdown_penalty: float = 2.0):
        super().__init__()
        self.dd_penalty = drawdown_penalty
    
    def forward(self, 
                predictions: torch.Tensor, 
                returns: torch.Tensor) -> torch.Tensor:
        
        positions = torch.tanh(predictions)
        portfolio_returns = positions * returns
        
        # Compute cumulative returns
        cumulative = torch.cumprod(1 + portfolio_returns.flatten(), dim=0)
        
        # Running maximum
        running_max = torch.cummax(cumulative, dim=0)[0]
        
        # Drawdown at each point
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = drawdowns.min()
        
        # Mean return
        mean_return = portfolio_returns.mean()
        
        # Combined loss: negative return + drawdown penalty
        return -mean_return + self.dd_penalty * max_drawdown.abs()


class ICLoss(nn.Module):
    """
    Loss based on Information Coefficient (rank correlation).
    
    Directly optimizes what we care about: ranking stocks correctly.
    """
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Soft ranking using sigmoid
        pred_ranks = torch.sigmoid(predictions * 10)  # Scale for better gradients
        target_ranks = torch.sigmoid(targets * 10)
        
        # Correlation loss (negative because we minimize)
        pred_centered = pred_ranks - pred_ranks.mean()
        target_centered = target_ranks - target_ranks.mean()
        
        numerator = (pred_centered * target_centered).sum()
        denominator = (pred_centered.norm() * target_centered.norm() + 1e-8)
        
        correlation = numerator / denominator
        
        return -correlation  # Negative because we want to maximize correlation


class CombinedLoss(nn.Module):
    """Combined loss for financial prediction with adjustable weights."""
    
    def __init__(self, mse_weight: float = 0.3, direction_weight: float = 0.5, ic_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.ic_weight = ic_weight
        self.direction_loss = DirectionalLoss()
        self.ic_loss = ICLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(predictions, targets)
        direction = self.direction_loss(predictions, targets)
        ic = self.ic_loss(predictions, targets)
        
        return self.mse_weight * mse + self.direction_weight * direction + self.ic_weight * ic


class CombinedFinancialLoss(nn.Module):
    """
    Production loss function combining multiple objectives.
    
    This is what you should actually use for training.
    """
    
    def __init__(self,
                 sharpe_weight: float = 1.0,
                 directional_weight: float = 0.5,
                 mse_weight: float = 0.1,
                 max_dd_weight: float = 0.3):
        super().__init__()
        
        self.sharpe_loss = SharpeRatioLoss()
        self.directional_loss = DirectionalLoss()
        self.max_dd_loss = MaxDrawdownLoss()
        
        self.weights = {
            'sharpe': sharpe_weight,
            'directional': directional_weight,
            'mse': mse_weight,
            'max_dd': max_dd_weight
        }
    
    def forward(self, 
                predictions: torch.Tensor, 
                returns: torch.Tensor) -> torch.Tensor:
        
        losses = {}
        
        losses['sharpe'] = self.sharpe_loss(predictions, returns)
        losses['directional'] = self.directional_loss(predictions, returns)
        losses['mse'] = F.mse_loss(predictions, returns)
        losses['max_dd'] = self.max_dd_loss(predictions, returns)
        
        total_loss = sum(
            self.weights[k] * v for k, v in losses.items()
        )
        
        return total_loss


def get_financial_loss(loss_type: str = 'combined') -> nn.Module:
    """
    Factory function for financial loss functions.
    
    Usage:
        loss_fn = get_financial_loss('sharpe')
        loss = loss_fn(predictions, returns)
    
    Available loss types:
        - 'mse': Standard MSE (DON'T use for trading)
        - 'sharpe': Sharpe ratio optimization
        - 'sortino': Sortino ratio (downside-focused)
        - 'directional': Penalizes wrong direction 3x
        - 'asymmetric': Quantile loss for asymmetric errors
        - 'max_dd': Penalizes drawdowns
        - 'ic': Information coefficient (rank correlation)
        - 'combined': Multi-objective (RECOMMENDED)
        - 'combined_financial': Full production loss
    """
    losses = {
        'mse': nn.MSELoss(),  # Don't use this for trading
        'sharpe': SharpeRatioLoss(),
        'sortino': SortinoRatioLoss(),
        'directional': DirectionalLoss(),
        'asymmetric': AsymmetricLoss(),
        'max_dd': MaxDrawdownLoss(),
        'ic': ICLoss(),
        'combined': CombinedLoss(),  # Use this
        'combined_financial': CombinedFinancialLoss(),
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(losses.keys())}")
    
    return losses[loss_type]
