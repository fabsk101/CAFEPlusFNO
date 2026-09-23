"""Self-contained CAFE+FNO ablation package.

Copy this directory as ``<CAFEPlusFNO repository>/ablations``.  No existing
repository source file needs to be changed.
"""

from .contracts import (
    ABLATION_CONDITIONS,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    MODEL_VARIANTS,
)

__all__ = [
    "ABLATION_CONDITIONS",
    "DEFAULT_SEEDS",
    "EXPERIMENT_ID",
    "MODEL_VARIANTS",
]

