"""Streaming sieve GMM for nonlinear instrumental variable regression."""

from .basis import feature_dimension, hermite_features
from .estimator import SSGMM
from .models import BaseModel, ExponentialModel, LinearModel, QuadraticModel

__version__ = "0.1.0"
__all__ = [
    "SSGMM",
    "BaseModel",
    "LinearModel",
    "QuadraticModel",
    "ExponentialModel",
    "hermite_features",
    "feature_dimension",
]
