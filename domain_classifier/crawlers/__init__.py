"""Crawler module for domain classifier.

This module contains website content crawlers that extract text from domains.
"""

# Make crawler functions available at package level
from domain_classifier.crawlers.apify_crawler import crawl_website
from domain_classifier.crawlers.direct_crawler import direct_crawl
