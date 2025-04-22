"""Domain utility functions for processing and analyzing domain names."""
import re
import logging
from urllib.parse import urlparse
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

def extract_domain_from_email(email: str) -> Optional[str]:
    """
    Extract domain from an email address.
    
    Args:
        email (str): The email address to process
        
    Returns:
        Optional[str]: The extracted domain or None if invalid
    """
    try:
        # Simple validation of email format
        if not email or '@' not in email:
            logger.warning(f"Invalid email format (missing @): {email}")
            return None
            
        # Extract domain portion (after @)
        domain = email.split('@')[-1].strip().lower()
        
        # Basic validation of domain
        if not domain or '.' not in domain:
            logger.warning(f"Invalid domain in email address: {email}")
            return None
            
        logger.info(f"Extracted domain '{domain}' from email '{email}'")
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from email: {e}")
        return None

def normalize_domain(domain: str) -> str:
    """
    Normalize a domain name by removing www prefix and ensuring lowercase.
    
    Args:
        domain (str): The domain to normalize
        
    Returns:
        str: The normalized domain
    """
    # Remove http/https protocol if present
    domain = domain.replace('https://', '').replace('http://', '')
    
    # Remove www prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
        
    # Convert to lowercase
    domain = domain.lower()
    
    # Remove trailing slash if present
    if domain.endswith('/'):
        domain = domain[:-1]
        
    return domain

def extract_domain_from_url(url: str) -> Optional[str]:
    """
    Extract domain from a URL.
    
    Args:
        url (str): The URL to process
        
    Returns:
        Optional[str]: The extracted domain or None if invalid
    """
    try:
        # Format URL properly if no protocol specified
        if not url.startswith('http'):
            url = 'https://' + url
            
        # Parse URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # If netloc is empty, try path (for cases like "example.com")
        if not domain:
            domain = parsed_url.path
            
        # Normalize domain
        domain = normalize_domain(domain)
        
        if not domain or '.' not in domain:
            logger.warning(f"Invalid domain in URL: {url}")
            return None
            
        logger.info(f"Extracted domain '{domain}' from URL '{url}'")
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from URL: {e}")
        return None

def is_valid_domain(domain: str) -> bool:
    """
    Check if a domain name is syntactically valid.
    
    Args:
        domain (str): The domain to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Basic domain validation pattern
    domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    
    return bool(re.match(domain_pattern, domain))

def get_normalized_url(domain: str) -> str:
    """
    Create a normalized URL from a domain.
    
    Args:
        domain (str): The domain name
        
    Returns:
        str: The normalized URL with https protocol
    """
    # Normalize domain first
    domain = normalize_domain(domain)
    
    # Return with https protocol
    return f"https://{domain}"
