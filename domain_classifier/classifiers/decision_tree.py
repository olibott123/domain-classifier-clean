import logging
import re
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def is_parked_domain(content: str) -> bool:
    """
    Enhanced detection of truly parked domains vs. just having minimal content.
    
    Args:
        content: The website content
        
    Returns:
        bool: True if the domain is parked/inactive
    """
    if not content:
        logger.info("Domain has no content at all, considering as parked")
        return True
        
    # Check for common parking phrases that indicate a domain is truly parked
    parking_phrases = [
        "domain is for sale", "buy this domain", "purchasing this domain", 
        "domain may be for sale", "this domain is for sale", "parked by",
        "domain parking", "this web page is parked", "website coming soon",
        "under construction", "site not published", "domain for sale",
        "under development", "this website is for sale"
    ]
    
    content_lower = content.lower()
    
    # Direct indicators of parked domains
    for phrase in parking_phrases:
        if phrase in content_lower:
            logger.info(f"Domain contains parking phrase: '{phrase}'")
            return True
    
    # Look for JavaScript-heavy sites with minimal crawled content
    if len(content.strip()) < 100:
        # More careful analysis before declaring parked
        
        # Check for common JS frameworks in the content
        js_indicators = ["react", "angular", "vue", "javascript", "script", "bootstrap", "jquery"]
        for indicator in js_indicators:
            if indicator in content_lower:
                logger.info(f"Found JS framework indicator: {indicator} - may be JS-heavy site, not parked")
                return False
        
        # Check for technical content that suggests a real site with poor crawling
        tech_indicators = ["<!doctype", "<html", "<head", "<meta", "<title", "<body", "<div"]
        tech_count = sum(1 for indicator in tech_indicators if indicator in content_lower)
        
        if tech_count >= 3:
            logger.info(f"Found {tech_count} HTML structure indicators - likely a real site with crawling issues")
            return False
        
        # Very little content with no indicators of real site structure
        if len(content.strip()) < 80:
            logger.info(f"Domain has extremely little content ({len(content.strip())} chars), considering as parked")
            return True
    
    # Very few words might indicate a parked domain, but be cautious
    words = re.findall(r'\b\w+\b', content_lower)
    unique_words = set(words)
    
    # An active site would typically have more unique words unless it's truly minimal
    if len(unique_words) < 15 and len(content.strip()) < 150:
        logger.info(f"Domain has very few unique words ({len(unique_words)}) and minimal content, considering as parked")
        return True
        
    return False

def check_special_domain_cases(domain: str, text_content: str) -> Optional[Dict[str, Any]]:
    """
    Check for special domain cases that need custom handling.
    
    Args:
        domain: The domain name
        text_content: The website content
        
    Returns:
        Optional[Dict[str, Any]]: Custom result if special case, None otherwise
    """
    domain_lower = domain.lower()
    
    # Check for special domains with known classifications
    # HostiFi - always MSP
    if "hostifi" in domain_lower:
        logger.info(f"Special case handling for known MSP domain: {domain}")
        return {
            "processing_status": 2,
            "is_service_business": True,
            "predicted_class": "Managed Service Provider",
            "internal_it_potential": 0,
            "confidence_scores": {
                "Managed Service Provider": 85,
                "Integrator - Commercial A/V": 8,
                "Integrator - Residential A/V": 5,
                "Internal IT Department": 0
            },
            "llm_explanation": f"{domain} is a cloud hosting platform specializing in Ubiquiti network management. They provide managed hosting services for UniFi Controller, UISP, and other network management software, which is a clear indication they are a Managed Service Provider focused on network infrastructure management.",
            "company_description": f"{domain} is a cloud hosting platform specializing in Ubiquiti network management and UniFi Controller hosting services.",
            "detection_method": "domain_override",
            "low_confidence": False,
            "max_confidence": 0.85
        }
        
    # Special handling for ciao.dk (known problematic vacation rental site)
    if domain_lower == "ciao.dk":
        logger.warning(f"Special handling for known vacation rental domain: {domain}")
        return {
            "processing_status": 2,
            "is_service_business": False,
            "predicted_class": "Internal IT Department",
            "internal_it_potential": 40,
            "confidence_scores": {
                "Managed Service Provider": 5,
                "Integrator - Commercial A/V": 3,
                "Integrator - Residential A/V": 2,
                "Internal IT Department": 40
            },
            "llm_explanation": f"{domain} appears to be a vacation rental/travel booking website offering holiday accommodations in various destinations. This is not a service business in the IT or A/V integration space. It's a travel industry business that might have a small internal IT department to maintain their booking systems and website.",
            "company_description": f"{domain} is a vacation rental and travel booking website offering holiday accommodations in various destinations.",
            "detection_method": "domain_override",
            "low_confidence": False,
            "max_confidence": 0.4
        }
        
    # Check for other vacation/travel-related domains
    vacation_terms = ["vacation", "holiday", "rental", "booking", "hotel", "travel", "accommodation", "ferie"]
    found_terms = [term for term in vacation_terms if term in domain_lower]
    if found_terms:
        logger.warning(f"Domain {domain} contains vacation/travel terms: {found_terms}")
        
        # Look for confirmation in the content
        travel_content_terms = ["booking", "accommodation", "stay", "vacation", "holiday", "rental"]
        if any(term in text_content.lower() for term in travel_content_terms):
            logger.warning(f"Content confirms {domain} is likely a travel/vacation site")
            return {
                "processing_status": 2,
                "is_service_business": False,
                "predicted_class": "Internal IT Department",
                "internal_it_potential": 35,
                "confidence_scores": {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 3,
                    "Integrator - Residential A/V": 2,
                    "Internal IT Department": 35
                },
                "llm_explanation": f"{domain} appears to be a travel/vacation related business, not an IT or A/V service provider. The website focuses on accommodations, bookings, or vacation rentals rather than technology services or integration. This type of business might have minimal internal IT needs depending on its size.",
                "company_description": f"{domain} appears to be a travel or vacation rental business focused on accommodations and bookings.",
                "detection_method": "domain_override",
                "low_confidence": False,
                "max_confidence": 0.35
            }
            
    # Check for transportation/logistics companies
    transport_terms = ["trucking", "transport", "logistics", "shipping", "freight", "delivery", "carrier"]
    found_transport_terms = [term for term in transport_terms if term in domain_lower]
    if found_transport_terms:
        logger.warning(f"Domain {domain} contains transportation terms: {found_transport_terms}")
        # Look for confirmation in the content
        transport_content_terms = ["shipping", "logistics", "fleet", "trucking", "transportation", "delivery"]
        if any(term in text_content.lower() for term in transport_content_terms):
            logger.warning(f"Content confirms {domain} is likely a transportation/logistics company")
            return {
                "processing_status": 2,
                "is_service_business": False,
                "predicted_class": "Internal IT Department",
                "internal_it_potential": 60,
                "confidence_scores": {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 3,
                    "Integrator - Residential A/V": 2,
                    "Internal IT Department": 60
                },
                "llm_explanation": f"{domain} appears to be a transportation and logistics company, not an IT or A/V service provider. The website focuses on shipping, transportation, and logistics services rather than technology services or integration. This type of company typically has moderate internal IT needs to manage their operations and fleet management systems.",
                "company_description": f"{domain} is a transportation and logistics company providing shipping and freight services.",
                "detection_method": "domain_override",
                "low_confidence": False,
                "max_confidence": 0.6
            }
            
    return None

def create_process_did_not_complete_result(domain: str = None) -> Dict[str, Any]:
    """
    Create a standardized result for when processing couldn't complete.
    
    Args:
        domain: The domain name
        
    Returns:
        dict: Standardized process failure result
    """
    domain_name = domain or "Unknown domain"
    
    return {
        "processing_status": 0,
        "is_service_business": None,
        "predicted_class": "Process Did Not Complete",
        "internal_it_potential": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        },
        "llm_explanation": f"Classification process for {domain_name} could not be completed. This may be due to connection issues, invalid domain, or other technical problems.",
        "company_description": f"Unable to determine what {domain_name} does due to processing failure.",
        "detection_method": "process_failed",
        "low_confidence": True,
        "max_confidence": 0.0
    }

def create_parked_domain_result(domain: str = None) -> Dict[str, Any]:
    """
    Create a standardized result for parked domains.
    
    Args:
        domain: The domain name
        
    Returns:
        dict: Standardized parked domain result
    """
    domain_name = domain or "This domain"
    
    return {
        "processing_status": 1,
        "is_service_business": None,
        "predicted_class": "Parked Domain",
        "internal_it_potential": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        },
        "llm_explanation": f"{domain_name} appears to be a parked or inactive domain. No business-specific content was found to determine the company type. This may be a domain that is reserved but not yet in use, for sale, or simply under construction.",
        "company_description": f"{domain_name} appears to be a parked or inactive domain with no active business.",
        "detection_method": "parked_domain_detection",
        "low_confidence": True,
        "is_parked": True,
        "max_confidence": 0.0
    }
