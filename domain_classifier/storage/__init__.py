"""Storage module for domain classifier.
This module handles data persistence and retrieval operations.
"""
# Make key components available at package level
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.storage.cache_manager import get_cached_result, cache_result, process_cached_result
from domain_classifier.storage.operations import save_to_snowflake, save_to_vector_db
from domain_classifier.storage.result_processor import process_fresh_result

# Try to make vector DB available, but don't fail if it's not there
try:
    from domain_classifier.storage.vector_db import VectorDBConnector, PINECONE_AVAILABLE, ANTHROPIC_AVAILABLE
    
    def create_vector_db(api_key=None, index_name=None, environment=None):
        """Create a vector database connector instance."""
        return VectorDBConnector(api_key=api_key, index_name=index_name, environment=environment)
except ImportError:
    import logging
    logging.getLogger(__name__).warning("Vector DB module not available")
