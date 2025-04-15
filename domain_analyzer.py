import re
import logging

logger = logging.getLogger(__name__)

def is_parked_domain(content: str) -> bool:
    """
    Detect if a domain is truly parked vs just having minimal content.
    
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
    
    # Extremely minimal content (likely parked)
    if len(content.strip()) < 80:
        logger.info(f"Domain has extremely little content ({len(content.strip())} chars), considering as parked")
        return True
    
    # Very few words (likely parked)
    words = re.findall(r'\b\w+\b', content_lower)
    if len(words) < 15:
        logger.info(f"Domain has very few words ({len(words)}), considering as parked")
        return True
        
    return False

def detect_minimal_content(content: str) -> bool:
    """
    Detect if domain has minimal content.
    
    Args:
        content: The website content
        
    Returns:
        bool: True if the domain has minimal content
    """
    if not content or len(content.strip()) < 100:
        logger.info(f"Domain content is very short: {len(content) if content else 0} characters")
        return True
        
    # Count words in content
    words = re.findall(r'\b\w+\b', content.lower())
    unique_words = set(words)
    
    # Return true if few words or unique words
    if len(words) < 50:
        logger.info(f"Domain has few words ({len(words)}), likely minimal content")
        return True
        
    if len(unique_words) < 30:
        logger.info(f"Domain has few unique words ({len(unique_words)}), likely minimal content")
        return True
            
    return False

def analyze_domain_name(domain: str) -> dict:
    """
    Analyze domain name for classification hints.
    
    Args:
        domain: The domain name
        
    Returns:
        dict: Analysis results with hints for classification
    """
    if not domain:
        return {
            "msp_hint": 0,
            "commercial_av_hint": 0,
            "residential_av_hint": 0,
            "service_hint": 0,
            "vacation_hint": False
        }
        
    domain_lower = domain.lower()
    
    analysis = {
        "msp_hint": 0,
        "commercial_av_hint": 0,
        "residential_av_hint": 0,
        "service_hint": 0,
        "vacation_hint": False
    }
    
    # MSP related domain terms
    msp_terms = ["it", "tech", "computer", "service", "cloud", "cyber", "network", 
                "support", "wifi", "unifi", "hosting", "host", "fi", "net", "msp", 
                "solutions", "consult", "technology", "systems"]
                
    for term in msp_terms:
        if term in domain_lower:
            analysis["msp_hint"] += 1
            analysis["service_hint"] += 1
            
    # Commercial A/V related domain terms
    commercial_terms = ["av", "audio", "visual", "video", "comm", "business", 
                        "enterprise", "corp", "pro", "professional", "commercial", 
                        "integrate", "system"]
                        
    for term in commercial_terms:
        if term in domain_lower:
            analysis["commercial_av_hint"] += 1
            analysis["service_hint"] += 1
            
    # Residential A/V related domain terms
    residential_terms = ["home", "residential", "smart", "theater", "cinema", 
                        "house", "domestic", "custom"]
                        
    for term in residential_terms:
        if term in domain_lower:
            analysis["residential_av_hint"] += 1
            analysis["service_hint"] += 1
            
    # Vacation/travel related terms
    vacation_terms = ["vacation", "holiday", "rental", "booking", "hotel", "travel", 
                     "accommodation", "ferie", "resort", "stay", "tour"]
                     
    analysis["vacation_hint"] = any(term in domain_lower for term in vacation_terms)
    
    # If it's a vacation domain, reduce residential AV hint
    if analysis["vacation_hint"]:
        analysis["residential_av_hint"] = 0
        
    return analysis
