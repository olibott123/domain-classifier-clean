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
    logger.info(f"Determining final classification for domain: {result.get('domain')}")
    logger.info(f"Result keys: {list(result.keys())}")
    logger.info(f"Error type: {result.get('error_type')}")
    logger.info(f"Is parked: {result.get('is_parked')}")
    logger.info(f"Predicted class: {result.get('predicted_class')}")
    
    # Check for DNS resolution errors first
    if result.get("error_type") == "dns_error" or (isinstance(result.get("explanation", ""), str) and "DNS" in result.get("explanation", "")):
        logger.info(f"Classifying as No Website available due to error_type={result.get('error_type')}")
        return "7-No Website available"
        
    # Check for parked domains
    if result.get("is_parked", False) or result.get("predicted_class") == "Parked Domain":
        # Check if Apollo data is available
        has_apollo = bool(result.get("apollo_data") and any(result["apollo_data"].values()))
        logger.info(f"Domain is parked. Has Apollo data: {has_apollo}")
        if has_apollo:
            return "5-Parked Domain with partial enrichment"
        else:
            return "6-Parked Domain - no enrichment"
    
    # Check for service business types
    predicted_class = result.get("predicted_class", "")
    logger.info(f"Checking service business type: {predicted_class}")
    
    if predicted_class == "Managed Service Provider":
        return "1-MSP"
    elif predicted_class == "Internal IT Department":
        return "2-Internal IT"
    elif predicted_class == "Integrator - Commercial A/V":
        return "3-Commercial Integrator"
    elif predicted_class == "Integrator - Residential A/V":
        return "4-Residential Integrator"
    
    # Default for unknown or error cases
    logger.info(f"No specific classification matched, defaulting to 2-Internal IT")
    return "2-Internal IT"  # Default to Internal IT if we can't determine
