"""Crawler module for domain classifier.

This module contains website content crawlers that extract text from domains.
"""

# Make the function directly available at this level
from domain_classifier.crawlers.apify_crawler import crawl_website, detect_error_type
from domain_classifier.crawlers.direct_crawler import direct_crawl
