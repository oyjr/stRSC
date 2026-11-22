"""Reproducible-signal correlation (RSC) for spatial transcriptomics predicted from histology.

    import strsc
    counts = strsc.read_counts("sample.h5ad")
    result = strsc.rsc(counts, {"my_model": "predictions.csv"}, layout="spots")
    print(result.scales)
"""

__version__ = "0.4.0"

from .files import Counts, InputError, Prediction, read_counts, read_files, read_predictions  # noqa: E402
from .api import Result, ideal_check, regions, rsc  # noqa: E402

__all__ = ["Counts", "InputError", "Prediction", "Result", "ideal_check", "read_counts", "read_files",
           "read_predictions", "regions", "rsc", "__version__"]
