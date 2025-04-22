"""
Domain Classifier System

A modular system for classifying websites into various business types based on 
their content, with a focus on identifying Managed Service Providers (MSPs) and A/V Integrators.

This package follows a modular design with the following components:

- api: Flask API for handling HTTP requests
- classifiers: Domain classification algorithms and models
- crawlers: Website content crawling functionality
- storage: Data persistence and retrieval
- utils: Utility functions and helpers
- config: Configuration settings

For usage and documentation, refer to the README.md file.
"""

__version__ = "1.0.0"
__author__ = "Oliver Bottrill"

# Import key components to make them available at the package level
from domain_classifier.classifiers.llm_classifier import LLMClassifier
from domain_classifier.crawlers.apify_crawler import crawl_website
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.api.app import create_app

# Convenience function to create a classifier instance
def create_classifier(api_key=None, model="claude-3-haiku-20240307"):
    """
    Create a new LLM classifier instance.
    
    Args:
        api_key (str, optional): API key for the classifier
        model (str, optional): Model to use for classification
        
    Returns:
        LLMClassifier: A new classifier instance
    """
    return LLMClassifier(api_key=api_key, model=model)
