"""Direct crawler for domain classification as a fallback."""
import requests
import logging
import re
from urllib.parse import urlparse
from typing import Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)

def direct_crawl(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Directly crawl a website using a simple GET request as a fallback method.
    
    Args:
        url (str): The URL to crawl
        
    Returns:
        tuple: (content, (error_type, error_detail))
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
    """
    try:
        logger.info(f"Attempting direct crawl for {url}")
        
        # Ensure URL is properly formatted
        if not url.startswith('http'):
            url = 'https://' + url
            
        # Set up headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # Make the request with a reasonable timeout
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        # Get content type
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Handle different content types
        if 'text/html' in content_type or 'text/plain' in content_type or 'application/xhtml' in content_type:
            # Extract readable text by removing HTML tags
            html_content = response.text
            clean_text = re.sub(r'<script.*?>.*?</script>', ' ', html_content, flags=re.DOTALL)
            clean_text = re.sub(r'<style.*?>.*?</style>', ' ', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if clean_text and len(clean_text) > 100:
                logger.info(f"Direct crawl successful, got {len(clean_text)} characters")
                return clean_text, (None, None)
            else:
                logger.warning(f"Direct crawl returned minimal content: {len(clean_text)} characters")
                # Return even minimal content rather than nothing
                if clean_text:
                    return clean_text, (None, None)
        else:
            logger.warning(f"Direct crawl returned non-text content: {content_type}")
            return None, ("non_text_content", f"Website returned non-text content: {content_type}")
        
        return None, ("empty_content", "Website returned empty or insufficient content")
    
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error during direct crawl: {e}")
        if "certificate verify failed" in str(e):
            return None, ("ssl_invalid", "The website has an invalid SSL certificate")
        elif "certificate has expired" in str(e):
            return None, ("ssl_expired", "The website's SSL certificate has expired")
        else:
            return None, ("ssl_error", "The website has SSL certificate issues")
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error during direct crawl: {e}")
        return None, ("connection_error", "Could not establish a connection to the website")
    
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during direct crawl: {e}")
        return None, ("timeout", "The website took too long to respond")
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during direct crawl: {e}")
        status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else None
        
        if status_code == 403 or status_code == 401:
            return None, ("access_denied", "Access to the website was denied")
        elif status_code == 404:
            return None, ("not_found", "The requested page was not found")
        elif status_code and 500 <= status_code < 600:
            return None, ("server_error", "The website is experiencing server errors")
        else:
            return None, ("http_error", f"HTTP error {status_code}")
    
    except Exception as e:
        logger.error(f"Unexpected error during direct crawl: {e}")
        return None, ("unknown_error", f"An unexpected error occurred: {str(e)}")

def try_multiple_protocols(domain: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Try crawling a domain with different protocols (https, http) in case one fails.
    
    Args:
        domain (str): The domain to crawl (without protocol)
        
    Returns:
        tuple: Same as direct_crawl
    """
    # First try https
    clean_domain = domain.replace('https://', '').replace('http://', '')
    https_url = f"https://{clean_domain}"
    
    logger.info(f"Trying HTTPS protocol for {clean_domain}")
    content, (error_type, error_detail) = direct_crawl(https_url)
    
    # If https failed due to SSL or connection issues, try http
    if not content and error_type in ['ssl_invalid', 'ssl_expired', 'ssl_error', 'connection_error']:
        logger.info(f"HTTPS failed, trying HTTP protocol for {clean_domain}")
        http_url = f"http://{clean_domain}"
        return direct_crawl(http_url)
    
    return content, (error_type, error_detail)

def extract_text_from_html(html_content: str) -> str:
    """
    Extract readable text from HTML content.
    
    Args:
        html_content (str): HTML content
        
    Returns:
        str: Extracted text
    """
    # Remove scripts and style blocks
    text = re.sub(r'<script.*?>.*?</script>', ' ', html_content, flags=re.DOTALL)
    text = re.sub(r'<style.*?>.*?</style>', ' ', text, flags=re.DOTALL)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    
    # Replace common tags with spaces or newlines for better readability
    text = re.sub(r'<br[^>]*>', '\n', text)
    text = re.sub(r'<p[^>]*>', '\n\n', text)
    text = re.sub(r'<div[^>]*>', '\n', text)
    text = re.sub(r'<li[^>]*>', '\n- ', text)
    text = re.sub(r'<h[1-6][^>]*>', '\n\n', text)
    
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Replace entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up the result
    text = text.strip()
    
    return text
