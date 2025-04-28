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
        self.runner = CrawlerRunner()
        dispatcher.connect(self._crawler_results, signal=signals.item_scraped)

    def _crawler_results(self, item):
        self.results.append(item)

    class GenericSpider(scrapy.Spider):
        name = "generic_spider"
        custom_settings = {
            'DOWNLOAD_TIMEOUT': 180,  # 3 minutes max
            'DEPTH_LIMIT': 1,  # Only homepage and direct links
            'CONCURRENT_REQUESTS': 16,
            'LOG_LEVEL': 'INFO'
        }

        def __init__(self, url):
            self.start_urls = [url]

        def parse(self, response):
            paragraphs = response.xpath('//p//text()').getall()
            yield {'url': response.url, 'paragraphs': paragraphs}

            for href in response.xpath('//a/@href').getall():
                if href.startswith("mailto:"):
                    continue
                yield response.follow(href, callback=self.parse)

    @crochet.wait_for(timeout=180.0)
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
    Crawl a website using Scrapy.
    
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
        
        # Create crawler instance and scrape
        crawler = ScrapyCrawler()
        content = crawler.scrape(url)
        
        if content and len(content.strip()) > 100:
            logger.info(f"Scrapy crawl successful, got {len(content)} characters")
            return content, (None, None)
        else:
            logger.warning(f"Scrapy crawl returned minimal content: {len(content) if content else 0} characters")
            return None, ("minimal_content", "Website returned minimal or no content")
            
    except Exception as e:
        from domain_classifier.crawlers.apify_crawler import detect_error_type
        error_type, error_detail = detect_error_type(str(e))
        logger.error(f"Error in Scrapy crawler: {e} (Type: {error_type})")
        return None, (error_type, error_detail)
