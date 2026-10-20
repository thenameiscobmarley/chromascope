"""Drop-in aimer runtime: numpy only, no training code. See runtime.py for how to use it."""
from .runtime import AimOutput, Aimer

__all__ = ["Aimer", "AimOutput"]
