"""Error handling utilities for domain classification."""
import logging
import socket
from typing import Dict, Any, Tuple, Optional

# Import final classification utility (if we can - use try/except to avoid circular imports)
try:
    from domain_classifier.utils.final_classification import determine_final_classification
    HAS_FINAL_CLASSIFICATION = True
except ImportError:
    HAS_FINAL_CLASSIFICATION = False

# Set up logging
logger = logging.getLogger(__name__)

def detect_error_type(error_message: str) -> Tuple[str, str]:
    """
    Analyze error message to determine the specific type of error.
    
    Args:
        error_message (str): The error message string
        
    Returns:
        tuple: (error_type, detailed_message)
    """
    error_message = str(error_message).lower()
    
    # SSL Certificate errors
    if any(phrase in error_message for phrase in ['certificate has expired', 'certificate verify failed', 'ssl', 'cert']):
        if 'expired' in error_message:
            return "ssl_expired", "The website's SSL certificate has expired."
        elif 'verify failed' in error_message:
            return "ssl_invalid", "The website has an invalid SSL certificate."
        else:
            return "ssl_error", "The website has SSL certificate issues."
    
    # DNS resolution errors
    elif any(phrase in error_message for phrase in ['getaddrinfo failed', 'name or service not known', 'no such host', 'dns']):
        return "dns_error", "The domain could not be resolved. It may not exist or DNS records may be misconfigured."
    
    # Connection errors
    elif any(phrase in error_message for phrase in ['connection refused', 'connection timed out', 'connection error']):
        return "connection_error", "Could not establish a connection to the website. It may be down or blocking our requests."
    
    # 4XX HTTP errors
    elif any(phrase in error_message for phrase in ['403', 'forbidden', '401', 'unauthorized']):
        return "access_denied", "Access to the website was denied. The site may be blocking automated access."
    elif '404' in error_message or 'not found' in error_message:
        return "not_found", "The requested page was not found on this website."
    
    # 5XX HTTP errors
    elif any(phrase in error_message for phrase in ['500', '502', '503', '504', 'server error']):
        return "server_error", "The website is experiencing server errors."
    
    # Robots.txt or crawling restrictions
    elif any(phrase in error_message for phrase in ['robots.txt', 'disallowed', 'blocked by robots']):
        return "robots_restricted", "The website has restricted automated access in its robots.txt file."
    
    # Default fallback
    return "unknown_error", "An unknown error occurred while trying to access the website."

def check_domain_dns(domain: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a domain has valid DNS resolution with strict timeout.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        tuple: (has_dns, error_message)
    """
    try:
        # Remove protocol if present
        clean_domain = domain.replace('https://', '').replace('http://', '')
        
        # Remove path if present
        if '/' in clean_domain:
            clean_domain = clean_domain.split('/', 1)[0]
        
        # Set a very short timeout for DNS resolution
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(3.0)  # 3 seconds max
            
        # Try to resolve the domain
        logger.info(f"Checking DNS resolution for domain: {clean_domain}")
        socket.gethostbyname(clean_domain)
        logger.info(f"DNS resolution successful for domain: {clean_domain}")
        return True, None
    except socket.gaierror as e:
        logger.warning(f"DNS resolution failed for {domain}: {e}")
        return False, f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured."
    except socket.timeout as e:
        logger.warning(f"DNS resolution timed out for {domain}: {e}")
        return False, f"DNS resolution timed out for {domain}. Domain may not exist or DNS server is not responding."
    except Exception as e:
        logger.error(f"Unexpected error checking DNS for {domain}: {e}")
        return False, f"Error checking DNS for {domain}: {e}"
    finally:
        # Reset timeout to default
        socket.setdefaulttimeout(original_timeout)

def create_error_result(domain: str, error_type: Optional[str] = None, 
                        error_detail: Optional[str] = None, email: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized error response based on the error type.
    
    Args:
        domain (str): The domain being processed
        error_type (str, optional): The type of error detected
        error_detail (str, optional): Detailed explanation of the error
        email (str, optional): Email address if processing an email
        
    Returns:
        dict: Standardized error response
    """
    # Default error response
    error_result = {
        "domain": domain,
        "error": "Failed to crawl website",
        "predicted_class": "Unknown",
        "confidence_score": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        },
        "low_confidence": True,
        "is_crawl_error": True
    }
    
    # Add email if provided
    if email:
        error_result["email"] = email
    
    # Add error_type if provided
    if error_type:
        error_result["error_type"] = error_type
    
    # Default explanation
    explanation = f"We were unable to retrieve content from {domain}. This could be due to a server timeout or the website being unavailable. Without analyzing the website content, we cannot determine the company type."
    
    # Enhanced error handling based on error type
    if error_type:
        if error_type.startswith('ssl_'):
            explanation = f"We couldn't analyze {domain} because of SSL certificate issues. "
            if error_type == 'ssl_expired':
                explanation += f"The website's SSL certificate has expired. This is a security issue with the target website, not our classification service."
            elif error_type == 'ssl_invalid':
                explanation += f"The website has an invalid SSL certificate. This is a security issue with the target website, not our classification service."
            else:
                explanation += f"This is a security issue with the target website, not our classification service."
            
            error_result["is_ssl_error"] = True
            
        elif error_type == 'dns_error':
            explanation = f"We couldn't analyze {domain} because the domain could not be resolved. This typically means the domain doesn't exist or its DNS records are misconfigured."
            error_result["is_dns_error"] = True
            
        elif error_type == 'connection_error':
            explanation = f"We couldn't analyze {domain} because a connection couldn't be established. The website may be down, temporarily unavailable, or blocking our requests."
            error_result["is_connection_error"] = True
            
        elif error_type == 'access_denied':
            explanation = f"We couldn't analyze {domain} because access was denied. The website may be blocking automated access or requiring authentication."
            error_result["is_access_denied"] = True
            
        elif error_type == 'not_found':
            explanation = f"We couldn't analyze {domain} because the main page was not found. The website may be under construction or have moved to a different URL."
            error_result["is_not_found"] = True
            
        elif error_type == 'server_error':
            explanation = f"We couldn't analyze {domain} because the website is experiencing server errors. This is an issue with the target website, not our classification service."
            error_result["is_server_error"] = True
            
        elif error_type == 'robots_restricted':
            explanation = f"We couldn't analyze {domain} because the website restricts automated access. This is a policy set by the website owner."
            error_result["is_robots_restricted"] = True
            
        elif error_type == 'timeout':
            explanation = f"We couldn't analyze {domain} because the website took too long to respond. The website may be experiencing performance issues or temporarily unavailable."
            error_result["is_timeout"] = True
            
        # If we have a specific error detail, use it to enhance the explanation
        if error_detail:
            explanation += f" {error_detail}"
    
    error_result["explanation"] = explanation
    
    # Add final classification if possible
    if HAS_FINAL_CLASSIFICATION:
        # Import here to avoid circular imports
        from domain_classifier.utils.final_classification import determine_final_classification
        error_result["final_classification"] = determine_final_classification(error_result)
    else:
        # Default for DNS errors
        if error_type == "dns_error":
            error_result["final_classification"] = "0-NO DNS RESOLUTION"
        else:
            error_result["final_classification"] = "4-IT"  # Default fallback
    
    return error_result

def handle_request_exception(e: Exception, domain: str, email: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle exceptions raised during API requests.
    
    Args:
        e (Exception): The exception that was raised
        domain (str): The domain being processed
        email (str, optional): Email address if processing an email
        
    Returns:
        dict: Error response for the API
    """
    logger.error(f"Error processing request for {domain}: {e}")
    
    # Try to identify the error type if possible
    error_type, error_detail = detect_error_type(str(e))
    
    # Create standardized error response
    error_result = create_error_result(domain, error_type, error_detail, email)
    
    # Add the actual error message
    error_result["error"] = str(e)
    
    return error_result
