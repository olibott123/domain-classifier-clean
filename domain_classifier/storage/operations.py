"""Storage operations for domain classification."""
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
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
        
        # Also save to vector database
        try:
            save_to_vector_db(domain, url, content, classification)
        except Exception as e:
            logger.error(f"Error saving to vector DB (non-critical): {e}")
            
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}\n{traceback.format_exc()}")
        return False

def save_to_vector_db(domain: str, url: str, content: str, classification: Dict[str, Any], vector_db_conn=None):
    """
    Save domain classification data to vector database.
    
    Args:
        domain: The domain being classified
        url: The URL of the website
        content: The content of the website
        classification: The classification result
        vector_db_conn: Optional vector DB connector instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # If Pinecone or Anthropic aren't available, just skip silently
        try:
            from domain_classifier.storage.vector_db import PINECONE_AVAILABLE, ANTHROPIC_AVAILABLE
            if not PINECONE_AVAILABLE or not ANTHROPIC_AVAILABLE:
                logger.info(f"Pinecone or Anthropic not available, skipping vector storage for {domain}")
                return False
        except ImportError:
            logger.info(f"Vector DB module not available, skipping vector storage for {domain}")
            return False
            
        # If no connector is provided, import and create one
        if vector_db_conn is None:
            try:
                logger.info("Creating new VectorDBConnector instance...")
                from domain_classifier.storage.vector_db import VectorDBConnector
                vector_db_conn = VectorDBConnector()
                logger.info(f"Created new VectorDBConnector instance, connected: {vector_db_conn.connected}")
            except Exception as e:
                logger.error(f"Error creating VectorDBConnector: {e}")
                logger.error(traceback.format_exc())
                return False
                
        # Skip if not connected to vector DB
        if not getattr(vector_db_conn, 'connected', False):
            logger.info(f"Vector DB not connected, skipping vector storage for {domain}")
            return False
            
        # Prepare metadata from classification
        logger.info(f"Preparing metadata for vector storage: {domain}")
        metadata = {
            "domain": domain,
            "url": url,
            "predicted_class": classification.get('predicted_class', 'Unknown'),
            "confidence_score": float(classification.get('max_confidence', 0.5)),
            "is_service_business": classification.get('is_service_business', None),
            "internal_it_potential": classification.get('internal_it_potential', 0),
            "detection_method": classification.get('detection_method', 'unknown'),
            "low_confidence": classification.get('low_confidence', False),
            "is_parked": classification.get('is_parked', False),
            "classification_date": classification.get('classification_date', '')
        }
        
        # Add company description if available
        if 'company_description' in classification:
            metadata['company_description'] = classification['company_description']
            
        # Store the vectorized data
        logger.info(f"Saving to vector DB for {domain}")
        success = vector_db_conn.upsert_domain_vector(
            domain=domain,
            content=content,
            metadata=metadata
        )
        
        if success:
            logger.info(f"✅ Successfully stored {domain} in vector database")
        
        return success
    except Exception as e:
        logger.error(f"Error saving to vector DB: {e}")
        logger.error(traceback.format_exc())
        return False

def query_similar_domains(query_text: str, top_k: int = 5, filter=None, vector_db_conn=None):
    """
    Query for domains similar to the given text.
    
    Args:
        query_text: The text to find similar domains for
        top_k: The number of results to return
        filter: Optional filter criteria
        vector_db_conn: Optional vector DB connector instance
    
    Returns:
        list: List of similar domains with metadata
    """
    try:
        # If Pinecone or Anthropic aren't available, just return empty
        try:
            from domain_classifier.storage.vector_db import PINECONE_AVAILABLE, ANTHROPIC_AVAILABLE
            if not PINECONE_AVAILABLE or not ANTHROPIC_AVAILABLE:
                logger.info("Pinecone or Anthropic not available, cannot query similar domains")
                return []
        except ImportError:
            logger.info("Vector DB module not available, cannot query similar domains")
            return []
            
        # If no connector is provided, import and create one
        if vector_db_conn is None:
            try:
                from domain_classifier.storage.vector_db import VectorDBConnector
                vector_db_conn = VectorDBConnector()
            except Exception as e:
                logger.error(f"Error creating VectorDBConnector: {e}")
                logger.error(traceback.format_exc())
                return []
                
        # Skip if not connected to vector DB
        if not getattr(vector_db_conn, 'connected', False):
            logger.info("Vector DB not connected, cannot query similar domains")
            return []
            
        # Call the vector DB to get similar domains
        logger.info(f"Querying vector DB for domains similar to: {query_text[:50]}...")
        similar_domains = vector_db_conn.query_similar_domains(
            query_text=query_text,
            top_k=top_k,
            filter=filter
        )
        
        logger.info(f"Found {len(similar_domains)} similar domains")
        return similar_domains
    except Exception as e:
        logger.error(f"Error querying similar domains: {e}")
        logger.error(traceback.format_exc())
        return []
