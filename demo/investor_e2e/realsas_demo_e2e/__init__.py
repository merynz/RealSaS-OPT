"""Generic single-specimen RealSaS demo stack.

No specimen is bound at import time.  Learned parameters may later memorize one
selected specimen, but interfaces and execution code are specimen-agnostic.
"""

from .iris_demo_v1 import IrisDemoConfig, IrisDemoV1
from .geppetto_demo_v1 import GeppettoDemoConfig, GeppettoDemoV1
from .arachne_demo_v1 import ArachneDemoConfig, ArachneDemoV1

__all__ = [
    "IrisDemoConfig", "IrisDemoV1",
    "GeppettoDemoConfig", "GeppettoDemoV1",
    "ArachneDemoConfig", "ArachneDemoV1",
]
