"""Multi-crawler module for domain classification that tries Scrapy first, then Apify."""
import requests
import logging
import time
import re
import socket
from urllib.parse import urlparse
from typing import Tuple, Optional, Any

# Explicitly define what should be exported from this module
__all__ = ['crawl_website', 'detect_error_type', 'apify_crawl']

# Import settings for API keys
from domain_classifier.config.settings import APIFY_TASK_ID, APIFY_API_TOKEN

# Import Scrapy crawler
from domain_classifier.crawlers.scrapy_crawler import scrapy_crawl

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
    elif any(phrase in error_message for phrase in ['getaddrinfo failed', 'name or service not known', 'no such host']):
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

def crawl_website(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Crawl a website using Scrapy first, then falling back to Apify if needed.
    
    Args:
        url (str): The URL to crawl
        
    Returns:
        tuple: (content, (error_type, error_detail), crawler_type)
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
            - crawler_type: The type of crawler used ("scrapy", "apify", "direct", etc.)
    """
    try:
        logger.info(f"Starting crawl for {url}")
        
        # Parse domain for later use
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Quick DNS check before attempting full crawl
        try:
            socket.gethostbyname(domain)
        except socket.gaierror:
            logger.warning(f"Domain {domain} does not resolve - DNS error")
            return None, ("dns_error", "This domain does not exist or cannot be resolved"), None
        
        # First try Scrapy crawler
        logger.info(f"Attempting crawl with Scrapy for {url}")
        content, (error_type, error_detail) = scrapy_crawl(url)
        
        # Special handling for parked domains identified during Scrapy crawl
        if error_type == "is_parked":
            logger.info(f"Scrapy identified {domain} as a parked domain")
            from domain_classifier.classifiers.decision_tree import create_parked_domain_result
            return None, (error_type, error_detail), "scrapy_parked_domain"
        
        # If Scrapy crawler succeeded, return the content
        if content and len(content.strip()) > 100:
            logger.info(f"Scrapy crawl successful for {url}")
            return content, (None, None), "scrapy"
            
        # Check if we got any content at all that might indicate a parked domain
        if content and len(content.strip()) > 0:
            # Check for parked domain indicators in minimal content
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(content, domain):
                logger.info(f"Detected parked domain from minimal Scrapy content: {domain}")
                return None, ("is_parked", "Domain appears to be parked based on content analysis"), "scrapy_minimal_parked_domain"
        
        # Log the failure reason
        if error_type:
            logger.warning(f"Scrapy crawler failed with error type: {error_type}. Falling back to Apify.")
        else:
            logger.warning(f"Scrapy crawler returned insufficient content. Falling back to Apify.")
            
        # Fall back to Apify crawler
        logger.info(f"Attempting fallback crawl with Apify for {url}")
        content, (error_type, error_detail) = apify_crawl(url)
        
        # Check for parked domain indicators in Apify content
        if content:
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(content, domain):
                logger.info(f"Detected parked domain from Apify content: {domain}")
                return None, ("is_parked", "Domain appears to be parked based on content analysis"), "apify_parked_domain"
        
        if content and len(content.strip()) > 100:
            return content, (error_type, error_detail), "apify"
        elif content:
            # Got some content, but not much
            return content, (error_type, error_detail), "apify_minimal"
        else:
            # No content at all
            return None, (error_type, error_detail), None
            
    except Exception as e:
        error_type, error_detail = detect_error_type(str(e))
        logger.error(f"Error crawling website: {e} (Type: {error_type})")
        return None, (error_type, error_detail), None

def apify_crawl(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Crawl a website using Apify with improved multi-stage approach for JavaScript-heavy sites.
    
    Args:
        url (str): The URL to crawl
        
    Returns:
        tuple: (content, (error_type, error_detail))
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
    """
    try:
        logger.info(f"Starting Apify crawl for {url}")
        
        # Extract domain for parked domain checks
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Start the crawl with standard settings
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,
            "maxCrawlPages": 5,
            "timeoutSecs": 120
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            run_id = response.json()['data']['id']
        except Exception as e:
            logger.error(f"Error starting Apify crawl: {e}")
            return None, detect_error_type(str(e))
            
        # Wait for crawl to complete
        endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
        
        max_attempts = 12
        for attempt in range(max_attempts):
            logger.info(f"Checking Apify crawl results, attempt {attempt+1}/{max_attempts}")
            
            try:
                response = requests.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data:
                        combined_text = ' '.join(item.get('text', '') for item in data if item.get('text'))
                        
                        # Check if this is a parked domain before continuing
                        if combined_text:
                            from domain_classifier.classifiers.decision_tree import is_parked_domain
                            if is_parked_domain(combined_text, domain):
                                logger.info(f"Detected parked domain during Apify crawl: {domain}")
                                return None, ("is_parked", "Domain appears to be parked based on content analysis")
                        
                        if combined_text and len(combined_text.strip()) > 100:
                            logger.info(f"Apify crawl completed, got {len(combined_text)} characters of content")
                            return combined_text, (None, None)
                        elif combined_text:
                            logger.warning(f"Apify crawl returned minimal content: {len(combined_text)} characters")
                            # Continue trying, might get better results on next attempt
                else:
                    logger.warning(f"Received status code {response.status_code} when checking Apify crawl results")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout when checking Apify status (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"Error checking Apify status: {e}")
            
            # Stage 2: Try Puppeteer approach explicitly after a few normal attempts
            if attempt == 4:
                logger.info("Trying Puppeteer-based approach for JavaScript-heavy site...")
                
                # Create a new task run with Puppeteer-specific settings
                puppeteer_payload = {
                    "startUrls": [{"url": url}],
                    "maxCrawlPages": 3,
                    "crawlerType": "playwright:chrome",  # Force Chrome browser
                    "dynamicContentWaitSecs": 15,        # Longer wait for JS content
                    "waitForSelectorSecs": 10,           # Wait for DOM elements
                    "expandIframes": True,               # Get iframe content
                    "clickElementsCssSelector": "button, [aria-expanded='false']",  # Click expandable elements
                    "saveHtml": True,                    # Save raw HTML for backup
                    "proxyConfiguration": {
                        "useApifyProxy": True,
                        "apifyProxyGroups": ["RESIDENTIAL"]  # Use residential proxies
                    }
                }
                
                try:
                    puppeteer_response = requests.post(endpoint.split('dataset')[0] + "?token=" + APIFY_API_TOKEN, 
                                                       json=puppeteer_payload, 
                                                       headers=headers, 
                                                       timeout=30)
                    puppeteer_response.raise_for_status()
                    puppeteer_run_id = puppeteer_response.json()['data']['id']
                    
                    # Wait for Puppeteer crawl to complete (separate from main loop)
                    puppeteer_endpoint = f"https://api.apify.com/v2/actor-runs/{puppeteer_run_id}/dataset/items?token={APIFY_API_TOKEN}"
                    
                    # Give it time to start up
                    time.sleep(5)
                    
                    for p_attempt in range(8):  # Fewer attempts but longer waits
                        logger.info(f"Checking Puppeteer crawl results, attempt {p_attempt+1}/8")
                        
                        try:
                            p_response = requests.get(puppeteer_endpoint, timeout=15)
                            
                            if p_response.status_code == 200:
                                p_data = p_response.json()
                                
                                if p_data:
                                    # Try to get text content from all fields that might have it
                                    text_fields = []
                                    for item in p_data:
                                        if item.get('text'):
                                            text_fields.append(item.get('text', ''))
                                        # Also try to extract from HTML if text is minimal
                                        elif item.get('html') and (not item.get('text') or len(item.get('text', '')) < 100):
                                            # Simple HTML to text extraction
                                            html_text = re.sub(r'<[^>]+>', ' ', item.get('html', ''))
                                            html_text = re.sub(r'\s+', ' ', html_text).strip()
                                            if len(html_text) > 100:
                                                text_fields.append(html_text)
                                    
                                    puppeteer_text = ' '.join(text_fields)
                                    
                                    # Check for parked domain indicators
                                    if puppeteer_text:
                                        from domain_classifier.classifiers.decision_tree import is_parked_domain
                                        if is_parked_domain(puppeteer_text, domain):
                                            logger.info(f"Detected parked domain during Puppeteer crawl: {domain}")
                                            return None, ("is_parked", "Domain appears to be parked based on content analysis")
                                    
                                    if puppeteer_text and len(puppeteer_text.strip()) > 100:
                                        logger.info(f"Puppeteer crawl successful, got {len(puppeteer_text)} characters")
                                        return puppeteer_text, (None, None)
                        except Exception as e:
                            logger.warning(f"Error checking Puppeteer crawl: {e}")
                            
                        time.sleep(12)  # Longer wait between Puppeteer checks
                        
                except Exception as e:
                    logger.error(f"Error with Puppeteer approach: {e}")
            
            # Stage 3: Direct request fallback (using direct_crawler module)
            if attempt == 7:  # After about 70 seconds
                logger.info("Trying direct request fallback...")
                from domain_classifier.crawlers.direct_crawler import direct_crawl
                
                direct_text, (direct_error_type, direct_error_detail), crawler_type = direct_crawl(url)
                
                # Check for parked domain indicators in direct crawl content
                if direct_text:
                    from domain_classifier.classifiers.decision_tree import is_parked_domain
                    if is_parked_domain(direct_text, domain):
                        logger.info(f"Detected parked domain during direct crawl: {domain}")
                        return None, ("is_parked", "Domain appears to be parked based on content analysis")
                
                if direct_text:
                    logger.info(f"Direct request successful, got {len(direct_text)} characters")
                    return direct_text, (None, None)
            
            # Wait between attempts for the main crawl
            if attempt < max_attempts - 1:
                time.sleep(10)
        
        # If we still don't have good content but have some minimal content,
        # return it rather than failing completely
        if 'combined_text' in locals() and combined_text:
            logger.warning(f"Using minimal content ({len(combined_text)} chars) as fallback")
            return combined_text, (None, None)
            
        # Try one last direct request if we have nothing else
        from domain_classifier.crawlers.direct_crawler import direct_crawl
        final_text, _, crawler_type = direct_crawl(url)
        
        # Check for parked domain indicators in final direct crawl
        if final_text:
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(final_text, domain):
                logger.info(f"Detected parked domain during final direct crawl: {domain}")
                return None, ("is_parked", "Domain appears to be parked based on content analysis")
                
        if final_text:
            logger.info(f"Final direct request got {len(final_text)} characters")
            return final_text, (None, None)
            
        logger.warning(f"Crawl timed out after all attempts")
        return None, ("timeout", "The website took too long to respond or has minimal crawlable content.")
    except Exception as e:
        error_type, error_detail = detect_error_type(str(e))
        logger.error(f"Error crawling with Apify: {e} (Type: {error_type})")
        return None, (error_type, error_detail)
