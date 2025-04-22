"""Custom JSON encoder for the domain classifier."""
import json
import logging
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle problematic types like NumPy values."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
