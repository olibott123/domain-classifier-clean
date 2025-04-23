"""Enrichment module for domain classifier.
This module adds additional company data to classification results.
"""
# Make key components available at the package level
from domain_classifier.enrichment.apollo_connector import ApolloConnector
from domain_classifier.enrichment.recommendation_engine import DomotzRecommendationEngine
from domain_classifier.enrichment.description_enhancer import enhance_company_description, generate_detailed_description
