"""Storage module for domain classifier.
This module handles data persistence and retrieval operations.
"""
# Make key components available at package level
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.storage.cache_manager import get_cached_result, cache_result, process_cached_result
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.storage.result_processor import process_fresh_result
