"""Scrapy-based crawler for domain classification."""
import logging
import crochet
crochet.setup()
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy import signals
from scrapy.signalmanager import dispatcher
from typing import Tuple, Optional
from urllib.parse import urlparse

# Set up logging
logger = logging.getLogger(__name__)

class ScrapyCrawler:
    def __init__(self):
        self.results = []
        self.runner = CrawlerRunner({
            # Add these settings to increase reliability
            'HTTPERROR_ALLOW_ALL': True,
            'DOWNLOAD_FAIL_ON_DATALOSS': False,
            'COOKIES_ENABLED': False,
            'RETRY_ENABLED': True,
            'RETRY_TIMES': 2,
            'DOWNLOAD_TIMEOUT': 30,
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'DOWNLOADER_CLIENTCONTEXTFACTORY': 'scrapy.core.downloader.contextfactory.BrowserLikeContextFactory'
        })
        dispatcher.connect(self._crawler_results, signal=signals.item_scraped)

    def _crawler_results(self, item):
        self.results.append(item)

    class GenericSpider(scrapy.Spider):
        name = "generic_spider"
        custom_settings = {
            'DEPTH_LIMIT': 1,  # Only homepage and direct links
            'LOG_LEVEL': 'INFO'
        }

        def __init__(self, url):
            self.start_urls = [url]

        def parse(self, response):
            try:
                # Try multiple extraction methods for better coverage
                paragraphs = []
                
                # Method 1: Extract paragraph text
                p_texts = response.xpath('//p//text()').getall()
                if p_texts:
                    paragraphs.extend(p_texts)
                
                # Method 2: Extract headings if we don't have enough text
                if len(' '.join(paragraphs)) < 100:
                    h_texts = response.xpath('//h1//text() | //h2//text() | //h3//text()').getall()
                    paragraphs.extend(h_texts)
                
                # Method 3: Extract div text if still not enough
                if len(' '.join(paragraphs)) < 200:
                    div_texts = response.xpath('//div[string-length(normalize-space(string())) > 20]//text()').getall()
                    paragraphs.extend(div_texts)
                
                # Method 4: Last resort - just get any visible text
                if len(' '.join(paragraphs)) < 300:
                    body_texts = response.xpath('//body//text()[string-length(normalize-space()) > 5]').getall()
                    paragraphs.extend(body_texts)
                
                # Clean up the text
                paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
                
                yield {'url': response.url, 'paragraphs': paragraphs}
                
                # Only follow a few links from homepage to avoid wasting resources
                if response.url == self.start_urls[0]:
                    for href in response.xpath('//a/@href').getall()[:10]:  # Limit to 10 links
                        if href.startswith("mailto:") or href.startswith("tel:"):
                            continue
                        yield response.follow(href, callback=self.parse)
                        
            except Exception as e:
                self.logger.error(f"Error parsing {response.url}: {e}")
                # Don't fail completely, just return what we have
                yield {'url': response.url, 'paragraphs': paragraphs if 'paragraphs' in locals() else []}

    @crochet.wait_for(timeout=60.0)  # Reduced timeout to avoid hanging
    def _run_spider(self, url):
        return self.runner.crawl(ScrapyCrawler.GenericSpider, url=url)

    def scrape(self, url):
        self.results.clear()
        try:
            self._run_spider(url)
            text = "\n".join(
                paragraph
                for item in self.results
                for paragraph in item.get('paragraphs', [])
            )
            return text
        except Exception as e:
            logger.error(f"Error in Scrapy crawler: {e}")
            return None


def scrapy_crawl(url: str) -> Tuple[Optional[str], Tuple[Optional[str], Optional[str]]]:
    """
    Crawl a website using Scrapy with better error handling.
    
    Args:
        url (str): The URL to crawl
        
    Returns:
        tuple: (content, (error_type, error_detail))
            - content: The crawled content or None if failed
            - error_type: Type of error if failed, None if successful
            - error_detail: Detailed error message if failed, None if successful
    """
    try:
        logger.info(f"Starting Scrapy crawl for {url}")
        
        # Parse the domain for parked domain checking later
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Create crawler instance and scrape
        crawler = ScrapyCrawler()
        content = crawler.scrape(url)
        
        # Check for parked domain indicators in content before proceeding
        if content:
            from domain_classifier.classifiers.decision_tree import is_parked_domain
            if is_parked_domain(content, domain):
                logger.info(f"Detected parked domain from Scrapy content: {domain}")
                return None, ("is_parked", "Domain appears to be parked based on content analysis"), None
                
            # Check for proxy errors or hosting provider mentions that indicate parked domains
            if len(content.strip()) < 300 and any(phrase in content.lower() for phrase in 
                                               ["proxy error", "error connecting", "godaddy", 
                                                "domain registration", "hosting provider"]):
                logger.info(f"Domain {domain} appears to be parked based on proxy errors or hosting mentions")
                return None, ("is_parked", "Domain appears to be parked with a domain registrar"), None
        
        if content and len(content.strip()) > 100:
            logger.info(f"Scrapy crawl successful, got {len(content)} characters")
            return content, (None, None)
        elif content:
            # Got some content but not much - might be enough for classification
            logger.warning(f"Scrapy crawl returned minimal content: {len(content)} characters")
            return content, (None, None)
        else:
            logger.warning(f"Scrapy crawl returned no content")
            return None, ("minimal_content", "Website returned minimal or no content")
            
    except Exception as e:
        from domain_classifier.crawlers.apify_crawler import detect_error_type
        error_type, error_detail = detect_error_type(str(e))
        logger.error(f"Error in Scrapy crawler: {e} (Type: {error_type})")
        return None, (error_type, error_detail)
