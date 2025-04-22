"""Storage operations for domain classification."""
import logging
import json
from typing import Dict, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

def save_to_snowflake(domain: str, url: str, content: str, classification: Dict[str, Any], 
                      snowflake_conn: Any) -> bool:
    """
    Save classification data to Snowflake.
    
    Args:
        domain (str): The domain name
        url (str): The URL that was crawled
        content (str): The website content
        classification (Dict[str, Any]): The classification result
        snowflake_conn: The Snowflake connector
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Always save the domain content
        logger.info(f"Saving content to Snowflake for {domain}")
        snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )

        # Ensure max_confidence exists
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence

        # Set low_confidence flag based on confidence threshold
        if 'low_confidence' not in classification:
            from domain_classifier.config.settings import LOW_CONFIDENCE_THRESHOLD
            classification['low_confidence'] = classification['max_confidence'] < LOW_CONFIDENCE_THRESHOLD

        # Get explanation directly from classification
        llm_explanation = classification.get('llm_explanation', '')
        
        # If explanation is too long, trim it properly at a sentence boundary
        if len(llm_explanation) > 4000:
            # Find the last period before 3900 chars
            last_period_index = llm_explanation[:3900].rfind('.')
            if last_period_index > 0:
                llm_explanation = llm_explanation[:last_period_index + 1]
            else:
                # If no period found, just truncate with an ellipsis
                llm_explanation = llm_explanation[:3900] + "..."
            
        # Create model metadata
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307'
        }
        
        # Convert model metadata to JSON string
        model_metadata_json = json.dumps(model_metadata)[:4000]  # Limit size
            
        # Special case for parked domains - save as "Parked Domain" if is_parked flag is set
        company_type = classification.get('predicted_class', 'Unknown')
        if classification.get('is_parked', False):
            company_type = "Parked Domain"
        
        logger.info(f"Saving classification to Snowflake: {domain}, {company_type}")
        snowflake_conn.save_classification(
            domain=domain,
            company_type=str(company_type),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification.get('confidence_scores', {}))[:4000],  # Limit size
            model_metadata=model_metadata_json,
            low_confidence=bool(classification.get('low_confidence', False)),
            detection_method=str(classification.get('detection_method', 'llm_classification')),
            llm_explanation=llm_explanation  # Add explanation directly to save_classification
        )
        
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}")
        return False

def process_fresh_result(classification: Dict[str, Any], domain: str, 
                         email: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a fresh classification result for API response.
    
    Args:
        classification (Dict[str, Any]): The classification result
        domain (str): The domain being classified
        email (Optional[str]): Optional email address
        url (Optional[str]): Optional website URL
        
    Returns:
        Dict[str, Any]: The processed result for API response
    """
    # Create a standardized result
    result = {
        "domain": domain,
        "predicted_class": classification.get("predicted_class", "Unknown"),
        "confidence_score": int(classification.get("max_confidence", 0.5) * 100),
        "confidence_scores": classification.get("confidence_scores", {}),
        "explanation": classification.get("llm_explanation", ""),
        "low_confidence": classification.get("low_confidence", False),
        "detection_method": classification.get("detection_method", "api"),
        "source": "fresh",
        "is_parked": classification.get("is_parked", False)
    }
    
    # Add email and URL if provided
    if email:
        result["email"] = email
    
    if url:
        result["website_url"] = url
    
    # Add company description if available
    if "company_description" in classification:
        result["company_description"] = classification["company_description"]
    
    return result

class FallbackSnowflakeConnector:
    """Fallback connector when Snowflake is not available."""
    
    def __init__(self):
        """Initialize the fallback connector."""
        self.connected = False
        logger.warning("Using FallbackSnowflakeConnector")
        
    def check_existing_classification(self, domain: str) -> Optional[Dict[str, Any]]:
        """Check existing classification (always returns None)."""
        logger.info(f"Fallback: No existing classification for {domain}")
        return None
        
    def save_domain_content(self, domain: str, url: str, content: str) -> Tuple[bool, Optional[str]]:
        """Save domain content (no-op)."""
        logger.info(f"Fallback: Not saving domain content for {domain}")
        return True, None
        
    def save_classification(self, domain: str, company_type: str, confidence_score: float, 
                           all_scores: str, model_metadata: str, low_confidence: bool, 
                           detection_method: str, llm_explanation: str) -> Tuple[bool, Optional[str]]:
        """Save classification (no-op)."""
        logger.info(f"Fallback: Not saving classification for {domain}")
        return True, None
        
    def get_domain_content(self, domain: str) -> Optional[str]:
        """Get domain content (always returns None)."""
        logger.info(f"Fallback: No content for {domain}")
        return None
        
    def close(self):
        """Close the connection (no-op)."""
        pass
