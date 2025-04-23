"""Global settings and configuration for the domain classifier."""
import os

# API keys and settings from environment variables
APIFY_TASK_ID = os.environ.get("APIFY_TASK_ID")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
APOLLO_API_KEY = os.environ.get("APOLLO_API_KEY")

# Configuration
LOW_CONFIDENCE_THRESHOLD = 0.7  # Threshold below which we consider a classification "low confidence"
AUTO_RECLASSIFY_THRESHOLD = 0.6  # Threshold below which we automatically reclassify

def get_port():
    """Get the port number from environment variables or use default."""
    return int(os.environ.get('PORT', 8080))
