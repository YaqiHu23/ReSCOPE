"""ReSCOPE: independent training and selection for 1, 2, and 3 ROIs."""
__version__ = "1.0.0"
from .api import ROI, RunResult, run_pipeline, select, train, validate
__all__ = ["ROI", "RunResult", "run_pipeline", "select", "train", "validate", "__version__"]
