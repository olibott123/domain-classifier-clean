"""Utility functions for determining final classification."""
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def determine_final_classification(result: Dict[str, Any]) -> str:
    """
    Determine the final classification based on the classification result.
    
    Args:
        result: The classification result
        
    Returns:
        str: The final classification code
    """
    # Check for DNS resolution errors first
    if result.get("error_type") == "dns_error" or "DNS" in result.get("explanation", ""):
        return "0-NO DNS RESOLUTION"
        
    # Check for parked domains
    if result.get("is_parked", False) or result.get("predicted_class") == "Parked Domain":
        # Check if Apollo data is available
        if result.get("apollo_data") and any(result["apollo_data"].values()):
            return "2-PARKED DOMAIN w Apollo"
        else:
            return "1-PARKED DOMAIN w/o Apollo"
    
    # Check for service business types
    predicted_class = result.get("predicted_class", "")
    
    if predicted_class == "Managed Service Provider":
        return "3-MSP"
    elif predicted_class == "Internal IT Department":
        return "4-IT"
    elif predicted_class == "Integrator - Commercial A/V":
        return "5-Commercial Integrator"
    elif predicted_class == "Integrator - Residential A/V":
        return "6-Residential Integrator"
    
    # Default for unknown or error cases
    return "4-IT"  # Default to IT if we can't determine
