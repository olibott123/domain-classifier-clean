"""Storage module for domain classifier.
This module handles data persistence and retrieval operations.
"""
# Make key components available at package level
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.storage.cache_manager import get_cached_result, cache_result, process_cached_result
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.storage.result_processor import process_fresh_result
# Add the vector database connector
from domain_classifier.storage.vector_db import VectorDBConnector

# Make vector DB available at package level for easy access
def create_vector_db(api_key=None, index_name=None):
    """Create a vector database connector instance."""
    return VectorDBConnector(api_key=api_key, index_name=index_name)
