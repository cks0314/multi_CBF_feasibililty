"""
Exact feasibility certification and optimal responsibility allocation for
multirobot control barrier function safety filters.

The package is deliberately small.  `model` builds the affine constraint
system from the fleet state, `certificate` evaluates the reserve and solves
for the best division of each shared constraint, and `filters` runs the
closed loop.
"""
from .model import (pairs, barrier_rows, nominal_input, step,
                    random_scenario, actuation_bounds)
from .certificate import (reserve, local_margin, margins_under, share_of,
                          alloc_uniform, alloc_capability, alloc_certificate,
                          ALLOCATIONS)
from .filters import LocalFilter, JointFilter, rollout

__all__ = [
    'pairs', 'barrier_rows', 'nominal_input', 'step', 'random_scenario',
    'actuation_bounds', 'reserve', 'local_margin', 'margins_under', 'share_of',
    'alloc_uniform', 'alloc_capability', 'alloc_certificate', 'ALLOCATIONS',
    'LocalFilter', 'JointFilter', 'rollout',
]
__version__ = '1.0.0'
