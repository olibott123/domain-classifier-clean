"""Domain override system for special cases."""
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Domain override system for special cases
DOMAIN_OVERRIDES = {
    # Format: 'domain': {'classification': 'type', 'confidence': score, 'explanation': 'text'}
    'nwaj.tech': {
        'classification': 'Managed Service Provider',
        'confidence': 85,
        'explanation': 'NWAJ Tech is a cybersecurity and zero trust security provider offering managed security services. They specialize in implementing zero trust security frameworks, which is a clear indication they are a Managed Service Provider focused on cybersecurity services.',
        'confidence_scores': {
            'Managed Service Provider': 85,
            'Integrator - Commercial A/V': 10,
            'Integrator - Residential A/V': 5
        },
        'company_description': 'NWAJ Tech is a cybersecurity provider specializing in zero trust security frameworks and managed security services for businesses.'
    },
    'hostifi.com': {
        'classification': 'Managed Service Provider',
        'confidence': 95,
        'explanation': 'HostiFi is a cloud hosting platform specializing in Ubiquiti network management. They provide managed hosting services for UniFi Controller, UISP, and other network management software, which is a clear indication they are a Managed Service Provider focused on network infrastructure management.',
        'confidence_scores': {
            'Managed Service Provider': 95,
            'Integrator - Commercial A/V': 3,
            'Integrator - Residential A/V': 2
        },
        'company_description': 'HostiFi is a cloud hosting platform providing managed hosting services for Ubiquiti network controllers including UniFi and UISP.'
    }
}

def check_domain_override(domain: str) -> Optional[Dict[str, Any]]:
    """
    Check if domain has a manual override classification.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        dict or None: Override classification if available, None otherwise
    """
    domain_lower = domain.lower()
    
    # Check exact match
    if domain_lower in DOMAIN_OVERRIDES:
        logger.info(f"Using override classification for {domain}")
        override = DOMAIN_OVERRIDES[domain_lower]
        
        # Create a standardized result
        result = {
            "domain": domain,
            "predicted_class": override['classification'],
            "confidence_score": override['confidence'],
            "confidence_scores": override.get('confidence_scores', {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0
            }),
            "explanation": override['explanation'],
            "company_description": override.get('company_description', 
                                              f"{domain} is a {override['classification']}."),
            "low_confidence": False,
            "detection_method": "manual_override",
            "source": "override",
            "is_parked": False,
            "max_confidence": override['confidence'] / 100.0
        }
        
        # Set service business flag
        result["is_service_business"] = result["predicted_class"] in [
            "Managed Service Provider", 
            "Integrator - Commercial A/V", 
            "Integrator - Residential A/V"
        ]
        
        # Ensure Internal IT Department score is included
        if "Internal IT Department" not in result["confidence_scores"]:
            if result["is_service_business"]:
                # For service businesses, Internal IT Department is always 0
                result["confidence_scores"]["Internal IT Department"] = 0
            else:
                # For non-service businesses, use internal_it_potential or default
                result["confidence_scores"]["Internal IT Department"] = override.get('internal_it_potential', 50)
                # Change the predicted class to "Internal IT Department"
                result["predicted_class"] = "Internal IT Department"
        
        return result
    
    # Check for domain pattern matches (for bulk overrides)
    for pattern, override in DOMAIN_OVERRIDES.items():
        if pattern.startswith('*.') and domain_lower.endswith(pattern[1:]):
            logger.info(f"Using pattern override classification for {domain} (matches {pattern})")
            
            # Create a standardized result (similar to above)
            result = {
                "domain": domain,
                "predicted_class": override['classification'],
                "confidence_score": override['confidence'],
                "confidence_scores": override.get('confidence_scores', {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0
                }),
                "explanation": override['explanation'],
                "company_description": override.get('company_description', 
                                                  f"{domain} is a {override['classification']}."),
                "low_confidence": False,
                "detection_method": "manual_override",
                "source": "override",
                "is_parked": False,
                "max_confidence": override['confidence'] / 100.0
            }
            
            # Set service business flag
            result["is_service_business"] = result["predicted_class"] in [
                "Managed Service Provider", 
                "Integrator - Commercial A/V", 
                "Integrator - Residential A/V"
            ]
            
            # Ensure Internal IT Department score is included
            if "Internal IT Department" not in result["confidence_scores"]:
                if result["is_service_business"]:
                    # For service businesses, Internal IT Department is always 0
                    result["confidence_scores"]["Internal IT Department"] = 0
                else:
                    # For non-service businesses, use internal_it_potential or default
                    result["confidence_scores"]["Internal IT Department"] = override.get('internal_it_potential', 50)
                    # Change the predicted class to "Internal IT Department"
                    result["predicted_class"] = "Internal IT Department"
            
            return result
    
    return None
