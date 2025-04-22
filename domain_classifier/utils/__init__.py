"""Utility module for domain classifier.

This module contains utility functions used across the domain classification system.
"""

# Make key utility functions available at the package level
from domain_classifier.utils.domain_utils import extract_domain_from_email
from domain_classifier.utils.error_handling import detect_error_type, create_error_result
from domain_classifier.utils.text_processing import extract_json, clean_json_string
