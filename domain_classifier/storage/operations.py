"""Storage operations for domain classification."""
import logging
import json
from typing import Dict, Any, Optional, Tuple
import traceback

# Set up logging
logger = logging.getLogger(__name__)

def save_to_snowflake(domain: str, url: str, content: str, classification: Dict[str, Any], snowflake_conn=None):
    """Save classification data to Snowflake"""
    try:
        # If no connector is provided, import and create one
        if snowflake_conn is None:
            from domain_classifier.storage.snowflake_connector import SnowflakeConnector
            snowflake_conn = SnowflakeConnector()
        
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
            # Import from config if available, otherwise use default
            try:
                from domain_classifier.config.settings import LOW_CONFIDENCE_THRESHOLD
                threshold = LOW_CONFIDENCE_THRESHOLD
            except ImportError:
                threshold = 0.7
                
            classification['low_confidence'] = classification['max_confidence'] < threshold

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
        logger.error(f"Error saving to Snowflake: {e}\n{traceback.format_exc()}")
        return False
