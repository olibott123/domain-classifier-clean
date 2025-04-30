"""Cache management for domain classification results."""
import logging
import json
from typing import Dict, Any, Optional

# Import final classification utility
from domain_classifier.utils.final_classification import determine_final_classification

# Set up logging
logger = logging.getLogger(__name__)

# In-memory cache for domain results (used when Snowflake is unavailable)
_domain_cache = {}

def cache_result(domain: str, result: Dict[str, Any]) -> None:
    """
    Cache a domain classification result in memory.
    
    Args:
        domain (str): The domain being classified
        result (Dict[str, Any]): The classification result
    """
    logger.info(f"Caching result for domain: {domain}")
    _domain_cache[domain.lower()] = result
    
def get_cached_result(domain: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached domain classification result.
    
    Args:
        domain (str): The domain to look up
        
    Returns:
        Optional[Dict[str, Any]]: The cached result or None if not found
    """
    domain_key = domain.lower()
    if domain_key in _domain_cache:
        logger.info(f"Found cached result for domain: {domain}")
        return _domain_cache[domain_key]
    logger.info(f"No cached result found for domain: {domain}")
    return None

def process_cached_result(record: Dict[str, Any], domain: str, email: Optional[str] = None, 
                          url: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a cached record from Snowflake to return a standardized result.
    
    Args:
        record (Dict[str, Any]): The record from Snowflake
        domain (str): The domain being classified
        email (Optional[str]): Optional email address
        url (Optional[str]): Optional website URL
        
    Returns:
        Dict[str, Any]: The processed result
    """
    logger.info(f"Processing cached record for domain: {domain}")
    
    # Extract confidence scores
    confidence_scores = {}
    try:
        confidence_scores = json.loads(record.get('ALL_SCORES', '{}'))
    except Exception as e:
        logger.warning(f"Could not parse ALL_SCORES for {domain}: {e}")
    
    # Extract LLM explanation from the LLM_EXPLANATION column
    llm_explanation = record.get('LLM_EXPLANATION', '')
    
    # If LLM_EXPLANATION is not available, try to get it from model_metadata
    if not llm_explanation:
        try:
            metadata = json.loads(record.get('MODEL_METADATA', '{}'))
            llm_explanation = metadata.get('llm_explanation', '')
        except Exception as e:
            logger.warning(f"Could not parse model_metadata for {domain}: {e}")
            
    # Extract Apollo data if available
    apollo_company_data = None
    apollo_person_data = None
    
    try:
        if record.get('APOLLO_COMPANY_DATA'):
            apollo_company_data = json.loads(record.get('APOLLO_COMPANY_DATA'))
            logger.info(f"Found cached Apollo company data for {domain}")
    except Exception as e:
        logger.warning(f"Could not parse APOLLO_COMPANY_DATA for {domain}: {e}")
    
    try:
        if record.get('APOLLO_PERSON_DATA'):
            apollo_person_data = json.loads(record.get('APOLLO_PERSON_DATA'))
            logger.info(f"Found cached Apollo person data for {domain}")
    except Exception as e:
        logger.warning(f"Could not parse APOLLO_PERSON_DATA for {domain}: {e}")
    
    # Ensure we have an explanation
    if not llm_explanation:
        llm_explanation = f"The domain {domain} was previously classified as a {record.get('COMPANY_TYPE')} based on analysis of website content."
    
    # Add low_confidence flag based on confidence score
    confidence_score = record.get('CONFIDENCE_SCORE', 0.5)
    low_confidence = record.get('LOW_CONFIDENCE', confidence_score < 0.7)
    
    # Check if it's a parked domain
    is_parked = record.get('COMPANY_TYPE') == "Parked Domain"
    
    # Handle legacy "Corporate IT" and "Non-Service Business"
    company_type = record.get('COMPANY_TYPE', 'Unknown')
    if company_type == "Non-Service Business":
        company_type = "Internal IT Department"
    
    # Create the standardized result
    result = {
        "domain": domain,
        "predicted_class": company_type,
        "confidence_score": int(confidence_score * 100),
        "confidence_scores": confidence_scores,
        "explanation": llm_explanation,
        "low_confidence": low_confidence,
        "detection_method": record.get('DETECTION_METHOD', 'api'),
        "source": "cached",
        "is_parked": is_parked
    }
    
    # Add crawler_type if present in record
    if record.get('CRAWLER_TYPE'):
        result["crawler_type"] = record.get('CRAWLER_TYPE')
        
    # Add classifier_type if present in record
    if record.get('CLASSIFIER_TYPE'):
        result["classifier_type"] = record.get('CLASSIFIER_TYPE')
    
    # Add Apollo data if available
    if apollo_company_data:
        result["apollo_data"] = apollo_company_data
    
    if apollo_person_data:
        result["apollo_person_data"] = apollo_person_data
    
    # Generate recommendations if Apollo data is available
    if apollo_company_data:
        try:
            from domain_classifier.enrichment.recommendation_engine import DomotzRecommendationEngine
            recommendation_engine = DomotzRecommendationEngine()
            recommendations = recommendation_engine.generate_recommendations(
                company_type, 
                apollo_company_data
            )
            result["domotz_recommendations"] = recommendations
            logger.info(f"Generated recommendations from cached Apollo data for {domain}")
        except Exception as e:
            logger.warning(f"Could not generate recommendations from cached data: {e}")
    
    # Add email and URL if provided
    if email:
        result["email"] = email
    
    if url:
        result["website_url"] = url
    
    # Add error_type if present in record
    if record.get('ERROR_TYPE'):
        result["error_type"] = record.get('ERROR_TYPE')
        
    # Determine and add the final classification
    result["final_classification"] = determine_final_classification(result)
    logger.info(f"Added final classification: {result['final_classification']} for {domain}")
    
    return result

def clear_cache():
    """Clear the in-memory cache."""
    global _domain_cache
    logger.info("Clearing in-memory cache")
    _domain_cache = {}
