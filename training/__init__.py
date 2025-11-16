"""Training modules for Deep CFR."""

from .deep_cfr import DeepCFR
from .trainer import Trainer
from .self_play import SelfPlayGenerator

__all__ = ['DeepCFR', 'Trainer', 'SelfPlayGenerator']

