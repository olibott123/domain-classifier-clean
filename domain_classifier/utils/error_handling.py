"""Error handling utilities for domain classification."""
import logging
import socket
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
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
    elif any(phrase in error_message for phrase in ['connection refused', 'connection timed out', 'connection error', 'connection reset', 'connection lost']):
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

def check_domain_dns(domain: str) -> Tuple[bool, Optional[str], bool]:
    """
    Check if a domain has valid DNS resolution AND can respond to a basic HTTP request.
    Also detects potentially flaky sites that may fail during crawling.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        tuple: (has_dns, error_message, potentially_flaky)
            - has_dns: Whether the domain has DNS resolution
            - error_message: Error message if DNS resolution failed
            - potentially_flaky: Whether the site shows signs of being flaky
    """
    potentially_flaky = False
    
    try:
        # Remove protocol if present
        clean_domain = domain.replace('https://', '').replace('http://', '')
        
        # Remove path if present
        if '/' in clean_domain:
            clean_domain = clean_domain.split('/', 1)[0]
        
        # Step 1: Try to resolve the domain using socket
        try:
            logger.info(f"Checking DNS resolution for domain: {clean_domain}")
            socket.setdefaulttimeout(3.0)  # 3 seconds max
            ip_address = socket.gethostbyname(clean_domain)
            logger.info(f"DNS resolution successful for domain: {clean_domain} (IP: {ip_address})")
            
            # Step 2: Try to establish a reliable HTTP connection
            try:
                logger.info(f"Attempting HTTP connection check for {clean_domain}")
                session = requests.Session()
                
                # Try HTTPS first
                success = False
                try:
                    url = f"https://{clean_domain}"
                    response = session.get(
                        url, 
                        timeout=5.0,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        },
                        stream=True
                    )
                    
                    # CRITICAL: Actually try to read a chunk of content
                    # This is what will detect connection issues with problematic sites
                    try:
                        chunk = next(response.iter_content(1024), None)
                        if chunk:
                            success = True
                            logger.info(f"Successfully read content chunk from {clean_domain}")
                        else:
                            potentially_flaky = True
                            logger.warning(f"No content received from {clean_domain}")
                    except Exception as read_error:
                        logger.warning(f"Error reading content from {clean_domain}: {read_error}")
                        potentially_flaky = True
                    
                    response.close()
                    
                    if success:
                        return True, None, False
                    
                except (ConnectionError, Timeout, RequestException) as https_e:
                    logger.warning(f"HTTPS failed for {clean_domain}, trying HTTP: {https_e}")
                    
                    # Look for reset indicators
                    if "ConnectionResetError" in str(https_e) or "reset by peer" in str(https_e):
                        potentially_flaky = True
                    
                    # Try HTTP as fallback
                    try:
                        url = f"http://{clean_domain}"
                        response = session.get(
                            url, 
                            timeout=5.0,
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                            },
                            stream=True
                        )
                        
                        # CRITICAL: Actually try to read a chunk of content
                        try:
                            chunk = next(response.iter_content(1024), None)
                            if chunk:
                                success = True
                                logger.info(f"Successfully read content chunk from {clean_domain} (HTTP)")
                            else:
                                potentially_flaky = True
                                logger.warning(f"No content received from {clean_domain} (HTTP)")
                        except Exception as read_error:
                            logger.warning(f"Error reading content from {clean_domain} (HTTP): {read_error}")
                            potentially_flaky = True
                        
                        response.close()
                        
                        if success:
                            return True, None, False
                        
                    except (ConnectionError, Timeout, RequestException) as http_e:
                        logger.warning(f"HTTP also failed for {clean_domain}: {http_e}")
                        
                        # Look for reset indicators
                        if "ConnectionResetError" in str(http_e) or "reset by peer" in str(http_e):
                            potentially_flaky = True
                            
                        # If it failed with both HTTPS and HTTP, it's not usable
                        error_message = f"The domain {domain} resolves but the web server is not responding properly. The server might be misconfigured or blocking requests."
                        
                        return False, error_message, potentially_flaky
                
                # If we got here, we tried both protocols but couldn't read content properly
                if potentially_flaky:
                    return False, f"The domain {domain} connects but fails during content transfer.", True
                
                return False, f"Could not establish a proper connection to {domain}.", False
                
            except Exception as conn_e:
                logger.warning(f"Connection error for {clean_domain}: {conn_e}")
                
                # Check for specific flaky indicators
                if "ConnectionResetError" in str(conn_e) or "reset by peer" in str(conn_e):
                    potentially_flaky = True
                    
                return False, f"The domain {domain} resolves but cannot be connected to. The server might be down or blocking connections.", potentially_flaky
                
        except socket.gaierror as e:
            logger.warning(f"DNS resolution failed for {domain}: {e}")
            return False, f"The domain {domain} could not be resolved. It may not exist or DNS records may be misconfigured.", False
            
    except socket.timeout as e:
        logger.warning(f"DNS resolution timed out for {domain}: {e}")
        return False, f"Timed out while checking {domain}. Domain may not exist or the server is not responding.", False
    except Exception as e:
        logger.error(f"Unexpected error checking domain {domain}: {e}")
        return False, f"Error checking {domain}: {e}", False

def is_domain_worth_crawling(domain: str) -> tuple:
    """
    Determines if a domain is worth attempting a full crawl based on preliminary checks.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        tuple: (worth_crawling, has_dns, error_msg, potentially_flaky)
    """
    has_dns, error_msg, potentially_flaky = check_domain_dns(domain)
    
    # Don't crawl if DNS resolution fails
    if not has_dns:
        logger.info(f"Domain {domain} failed DNS check: {error_msg}")
        return False, has_dns, error_msg, potentially_flaky
        
    # Be cautious with potentially flaky domains but still allow crawling
    if potentially_flaky:
        logger.warning(f"Domain {domain} may be flaky, proceeding with caution")
        
    return True, has_dns, error_msg, potentially_flaky

def create_error_result(domain: str, error_type: Optional[str] = None, 
                        error_detail: Optional[str] = None, email: Optional[str] = None,
                        crawler_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized error response based on the error type.
    
    Args:
        domain (str): The domain being processed
        error_type (str, optional): The type of error detected
        error_detail (str, optional): Detailed explanation of the error
        email (str, optional): Email address if processing an email
        crawler_type (str, optional): The type of crawler used/attempted
        
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
            
        elif error_type == 'is_parked':
            explanation = f"The domain {domain} appears to be parked or inactive. This domain may be registered but not actively in use for a business."
            error_result["is_parked"] = True
            error_result["predicted_class"] = "Parked Domain"
            
        # If we have a specific error detail, use it to enhance the explanation
        if error_detail:
            explanation += f" {error_detail}"
    
    error_result["explanation"] = explanation
    
    # Add crawler_type if provided
    error_result["crawler_type"] = crawler_type or "error_handler"  # Set a default
    
    # Add final classification if possible
    if HAS_FINAL_CLASSIFICATION:
        # Import here to avoid circular imports
        from domain_classifier.utils.final_classification import determine_final_classification
        error_result["final_classification"] = determine_final_classification(error_result)
    else:
        # Default for DNS errors or connection errors
        if error_type in ["dns_error", "connection_error"]:
            error_result["final_classification"] = "7-No Website available"
        elif error_type == "is_parked":
            error_result["final_classification"] = "6-Parked Domain - no enrichment"
        else:
            error_result["final_classification"] = "2-Internal IT"  # Default fallback
    
    return error_result
