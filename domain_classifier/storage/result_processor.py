"""Process classification results for API responses."""
import logging
import json
import re
from typing import Dict, Any, Optional

# Import final classification utility
from domain_classifier.utils.final_classification import determine_final_classification

# Set up logging
logger = logging.getLogger(__name__)

def process_fresh_result(classification: Dict[str, Any], domain: str, email: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a fresh classification result.
    
    Args:
        classification: The classification result from the classifier
        domain: The domain name
        email: Optional email address
        url: Optional URL for clickable link
        
    Returns:
        dict: The processed result ready for the client
    """
    try:
        if classification.get("is_parked", False):
            # Special case for parked domains
            result = {
                "domain": domain,
                "predicted_class": "Parked Domain",
                "confidence_score": 0,
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0
                },
                "explanation": classification.get('llm_explanation', 'This appears to be a parked or inactive domain without business-specific content.'),
                "company_description": classification.get('company_description', f"{domain} appears to be a parked or inactive domain with no active business."),
                "low_confidence": True,
                "detection_method": classification.get('detection_method', 'parked_domain_detection'),
                "source": "fresh",
                "is_parked": True
            }
        else:
            # Normal case with confidence scores as integers (1-100)
            # Get max confidence 
            max_confidence = 0
            if "max_confidence" in classification:
                if isinstance(classification["max_confidence"], float) and classification["max_confidence"] <= 1.0:
                    max_confidence = int(classification["max_confidence"] * 100)
                else:
                    max_confidence = int(classification["max_confidence"])
            else:
                # If max_confidence not set, find the highest score
                confidence_scores = classification.get('confidence_scores', {})
                if confidence_scores:
                    max_score = max(confidence_scores.values())
                    if isinstance(max_score, float) and max_score <= 1.0:
                        max_confidence = int(max_score * 100)
                    else:
                        max_confidence = int(max_score)
            
            # Get confidence scores with type handling
            processed_scores = {}
            for category, score in classification.get('confidence_scores', {}).items():
                # Convert float 0-1 to int 1-100
                if isinstance(score, float) and score <= 1.0:
                    processed_scores[category] = int(score * 100)
                # Already int in 1-100 range
                elif isinstance(score, (int, float)):
                    processed_scores[category] = int(score)
                # String (somehow)
                else:
                    try:
                        score_float = float(score)
                        if score_float <= 1.0:
                            processed_scores[category] = int(score_float * 100)
                        else:
                            processed_scores[category] = int(score_float)
                    except (ValueError, TypeError):
                        # Default if conversion fails
                        processed_scores[category] = 5
            
            # Handle legacy "Corporate IT" key
            if "Corporate IT" in processed_scores:
                score = processed_scores.pop("Corporate IT")
                processed_scores["Internal IT Department"] = score
                
            # Final validation - ensure scores are different
            if len(set(processed_scores.values())) <= 1:
                logger.warning("API response has identical confidence scores, fixing...")
                pred_class = classification.get('predicted_class')
                
                # Handle legacy "Non-Service Business" predicted class
                if pred_class == "Non-Service Business":
                    pred_class = "Internal IT Department"
                    
                if pred_class == "Managed Service Provider":
                    processed_scores = {
                        "Managed Service Provider": 90,
                        "Integrator - Commercial A/V": 10,
                        "Integrator - Residential A/V": 10,
                        "Internal IT Department": 0
                    }
                elif pred_class == "Integrator - Commercial A/V":
                    processed_scores = {
                        "Integrator - Commercial A/V": 90,
                        "Managed Service Provider": 10,
                        "Integrator - Residential A/V": 10,
                        "Internal IT Department": 0
                    }
                elif pred_class == "Integrator - Residential A/V":
                    processed_scores = {
                        "Integrator - Residential A/V": 90,
                        "Integrator - Commercial A/V": 10, 
                        "Managed Service Provider": 10,
                        "Internal IT Department": 0
                    }
                elif pred_class == "Process Did Not Complete":
                    # Set all scores to 0 for process_did_not_complete
                    processed_scores = {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    }
                    # Reset max_confidence to 0.0
                    max_confidence = 0
                elif pred_class == "Internal IT Department":
                    # For Internal IT Department, add Internal IT Department score
                    internal_it_potential = classification.get('internal_it_potential', 60)
                    if internal_it_potential is None:
                        internal_it_potential = 60
                        
                    processed_scores = {
                        "Managed Service Provider": 5,
                        "Integrator - Commercial A/V": 3,
                        "Integrator - Residential A/V": 2,
                        "Internal IT Department": internal_it_potential
                    }
                
                # Update max_confidence to match the new highest value if not Process Did Not Complete
                if pred_class not in ["Process Did Not Complete", "Internal IT Department"]:
                    max_confidence = 90
                    
            # Ensure explanation exists
            explanation = classification.get('llm_explanation', '')
            if not explanation:
                explanation = f"Based on analysis of website content, {domain} has been classified as a {classification.get('predicted_class')}."
                
            # Check for Non-Service Business in the explanation 
            if "non-service business" in explanation.lower() and classification.get('predicted_class') in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                if max_confidence <= 20:  # Only override if confidence is low
                    logger.info(f"Correcting classification for {domain} to Internal IT Department based on explanation")
                    classification['predicted_class'] = "Internal IT Department"

            # Handle legacy "Non-Service Business" predicted class
            if classification.get('predicted_class') == "Non-Service Business":
                classification['predicted_class'] = "Internal IT Department"
                
            # For Internal IT Department, ensure Internal IT Department score is included
            if classification.get('predicted_class') == "Internal IT Department":
                # Add Internal IT Department for Internal IT Department if not already present
                if "Internal IT Department" not in processed_scores:
                    internal_it_potential = classification.get('internal_it_potential', 60)
                    if internal_it_potential is None:
                        internal_it_potential = 60
                        
                    processed_scores["Internal IT Department"] = internal_it_potential
                    # Ensure service scores are low
                    for category in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                        processed_scores[category] = min(processed_scores.get(category, 5), 10)
                
                # Set confidence score to a consistent value for Internal IT Department
                max_confidence = 80
            
            # For service businesses, ensure Internal IT Department is 0
            elif classification.get('predicted_class') in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                processed_scores["Internal IT Department"] = 0
                
            # Create the final result
            result = {
                "domain": domain,
                "predicted_class": classification.get('predicted_class'),
                "confidence_score": max_confidence,
                "confidence_scores": processed_scores,
                "explanation": explanation,
                "company_description": classification.get('company_description', f"{domain} is a {classification.get('predicted_class')}."),
                "low_confidence": classification.get('low_confidence', False),
                "detection_method": classification.get('detection_method', 'api'),
                "source": "fresh",
                "is_parked": False
            }

        # Add website URL for clickable link if provided
        if url:
            result["website_url"] = url
            
        # Add email to response if provided
        if email:
            result["email"] = email
        
        # Add error_type if present in classification
        if "error_type" in classification:
            result["error_type"] = classification["error_type"]
            
        # Determine and add the final classification
        result["final_classification"] = determine_final_classification(result)
        logger.info(f"Added final classification: {result['final_classification']} for {domain}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing fresh result: {e}")
        # Return a basic result with error information
        error_result = {
            "domain": domain,
            "predicted_class": classification.get('predicted_class', 'Unknown'),
            "confidence_score": 50,
            "confidence_scores": {
                "Managed Service Provider": 10,
                "Integrator - Commercial A/V": 5,
                "Integrator - Residential A/V": 5,
                "Internal IT Department": 0
            },
            "explanation": f"We encountered an error processing the classification result for {domain}.",
            "company_description": f"{domain} is a business that we attempted to classify.",
            "low_confidence": True,
            "detection_method": "error_during_processing",
            "source": "error",
            "is_parked": False,
            "error": str(e),
            "final_classification": "4-IT"  # Default for error cases
        }
        return error_result
