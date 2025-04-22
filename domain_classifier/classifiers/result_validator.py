import logging
import re
from typing import Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

def validate_classification(classification: Dict[str, Any], domain: str = None) -> Dict[str, Any]:
    """
    Validate and normalize classification results.
    
    Args:
        classification: The classification to validate
        domain: Optional domain name for context
        
    Returns:
        dict: The validated classification
    """
    # Set default processing_status if not present
    if "processing_status" not in classification:
        classification["processing_status"] = 2  # Success
        
    # Check for parked domain
    if classification.get("processing_status") == 1:
        # This is a parked domain, no further validation needed
        classification["is_service_business"] = None
        classification["predicted_class"] = "Parked Domain"
        classification["internal_it_potential"] = 0
        classification["confidence_scores"] = {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        }
        classification["max_confidence"] = 0.0
        classification["low_confidence"] = True
        classification["is_parked"] = True
        return classification
        
    # Check for process failure
    if classification.get("processing_status") == 0:
        # Process did not complete, no further validation needed
        classification["is_service_business"] = None
        classification["predicted_class"] = "Process Did Not Complete"
        classification["internal_it_potential"] = 0
        classification["confidence_scores"] = {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Internal IT Department": 0
        }
        classification["max_confidence"] = 0.0
        classification["low_confidence"] = True
        return classification
    
    # Ensure required fields exist
    if "predicted_class" not in classification or classification["predicted_class"] is None:
        logger.warning("Missing or null predicted_class in classification, using fallback")
        
        # Check explanation for clues about what type of business this is
        explanation = classification.get("llm_explanation", "").lower()
        if "non-service business" in explanation or "not a service" in explanation:
            classification["predicted_class"] = "Internal IT Department"
        elif "travel" in explanation or "vacation" in explanation or "rental" in explanation:
            classification["predicted_class"] = "Internal IT Department"
        elif "transport" in explanation or "logistics" in explanation or "shipping" in explanation:
            classification["predicted_class"] = "Internal IT Department"
        else:
            classification["predicted_class"] = "Unknown"
        
        logger.info(f"Fixed null predicted_class to '{classification['predicted_class']}' based on explanation")
        
    if "is_service_business" not in classification:
        logger.warning("Missing is_service_business in classification, inferring from predicted_class")
        classification["is_service_business"] = classification["predicted_class"] in [
            "Managed Service Provider", 
            "Integrator - Commercial A/V", 
            "Integrator - Residential A/V"
        ]
        
    is_service = classification.get("is_service_business", True)
    
    # Check for very low confidence service business classifications
    if is_service and "confidence_scores" in classification:
        # Get highest confidence score
        highest_score = max(classification["confidence_scores"].values())
        if highest_score <= 15:
            # This is likely not actually a service business
            logger.warning(f"Very low confidence ({highest_score}) for service classification, recategorizing as Internal IT Department")
            classification["is_service_business"] = False
            classification["predicted_class"] = "Internal IT Department"
            is_service = False
            
            # Set appropriate internal IT potential
            if "llm_explanation" in classification:
                # Extract potential internal IT score from explanation
                it_match = re.search(r'internal IT.*?(\d+)[/\s]*100', classification["llm_explanation"])
                if it_match:
                    classification["internal_it_potential"] = int(it_match.group(1))
                else:
                    classification["internal_it_potential"] = 50  # Default medium value
    
    if "confidence_scores" not in classification:
        logger.warning("Missing confidence_scores in classification, using fallback")
        if is_service:
            classification["confidence_scores"] = {
                "Managed Service Provider": 50,
                "Integrator - Commercial A/V": 25,
                "Integrator - Residential A/V": 15,
                "Internal IT Department": 0
            }
        else:
            classification["confidence_scores"] = {
                "Managed Service Provider": 5,
                "Integrator - Commercial A/V": 3,
                "Integrator - Residential A/V": 2
            }
            
    if "internal_it_potential" not in classification:
        logger.warning("Missing internal_it_potential in classification, using fallback")
        if is_service:
            classification["internal_it_potential"] = 0
        else:
            # Default middle value for unknown
            classification["internal_it_potential"] = 50
    
    if "llm_explanation" not in classification or not classification["llm_explanation"]:
        logger.warning("Missing llm_explanation in classification, using fallback")
        if is_service:
            classification["llm_explanation"] = f"Based on the available information, this appears to be a {classification['predicted_class']}."
        else:
            classification["llm_explanation"] = f"This appears to be a non-service business. It doesn't provide IT or A/V integration services."
    
    # Normalize confidence scores
    confidence_scores = classification["confidence_scores"]
    
    # Check if scores need to be converted from 0-1 to 1-100 scale
    if any(isinstance(score, float) and 0 <= score <= 1 for score in confidence_scores.values()):
        logger.info("Converting confidence scores from 0-1 scale to 1-100")
        confidence_scores = {k: int(v * 100) for k, v in confidence_scores.items()}
    
    # Ensure all required categories exist
    required_categories = ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]
    for category in required_categories:
        if category not in confidence_scores:
            logger.warning(f"Missing category {category} in confidence scores, adding default")
            confidence_scores[category] = 5 if not is_service else 30
            
    # Ensure scores are within valid range (1-100)
    confidence_scores = {k: max(1, min(100, int(v))) for k, v in confidence_scores.items()}
    
    # For non-service businesses, ensure service scores are appropriately low
    if not is_service:
        for category in required_categories:
            if confidence_scores[category] > 10:
                logger.warning(f"Reducing {category} score for non-service business")
                confidence_scores[category] = min(confidence_scores[category], 10)
        
        # Ensure internal_it_potential is an integer
        if classification["internal_it_potential"] is not None:
            classification["internal_it_potential"] = int(classification["internal_it_potential"])
            
        # Add Internal IT Department score based on internal_it_potential
        it_potential = classification.get("internal_it_potential", 50)
        confidence_scores["Internal IT Department"] = it_potential
    else:
        # Add Internal IT Department score with value 0 for service businesses
        confidence_scores["Internal IT Department"] = 0
            
    # For service businesses, ensure scores are differentiated
    if is_service and (len(set(confidence_scores.values())) <= 1 or 
                        max(confidence_scores.values()) - min(confidence_scores.values()) < 5):
        logger.warning("Confidence scores not sufficiently differentiated for service business, adjusting them")
        
        pred_class = classification["predicted_class"]
        
        # Set base scores to ensure strong differentiation
        if pred_class == "Managed Service Provider":
            confidence_scores = {
                "Managed Service Provider": 80,
                "Integrator - Commercial A/V": 15,
                "Integrator - Residential A/V": 5,
                "Internal IT Department": 0  # Ensure Internal IT Department is always included with 0 for service businesses
            }
        elif pred_class == "Integrator - Commercial A/V":
            confidence_scores = {
                "Integrator - Commercial A/V": 80,
                "Managed Service Provider": 15,
                "Integrator - Residential A/V": 5,
                "Internal IT Department": 0  # Ensure Internal IT Department is always included with 0 for service businesses
            }
        else:  # Residential A/V
            confidence_scores = {
                "Integrator - Residential A/V": 80,
                "Integrator - Commercial A/V": 15,
                "Managed Service Provider": 5,
                "Internal IT Department": 0  # Ensure Internal IT Department is always included with 0 for service businesses
            }
    
    # For service businesses, ensure predicted class matches highest confidence category
    if is_service:
        highest_category = max(confidence_scores.items(), key=lambda x: x[1] if x[0] != "Internal IT Department" else 0)[0]
        if classification["predicted_class"] != highest_category:
            logger.warning(f"Predicted class {classification['predicted_class']} doesn't match highest confidence category {highest_category}, fixing")
            classification["predicted_class"] = highest_category
        
    # Calculate max confidence for consistency
    if is_service:
        classification["max_confidence"] = confidence_scores[classification["predicted_class"]] / 100.0
    else:
        # For non-service businesses, max confidence is based on internal IT potential certainty
        classification["max_confidence"] = 0.8 if classification["internal_it_potential"] is not None else 0.5
    
    # Add low_confidence flag based on highest score or other factors
    if is_service:
        classification["low_confidence"] = confidence_scores[classification["predicted_class"]] < 40
    else:
        # For non-service, we're less confident overall
        classification["low_confidence"] = True
        
    # Update the classification with validated scores
    classification["confidence_scores"] = confidence_scores
    
    # Ensure company_description is present
    if "company_description" not in classification:
        from domain_classifier.utils.text_processing import extract_company_description
        classification["company_description"] = extract_company_description(
            "", classification.get("llm_explanation", ""), domain
        )
    
    return classification


def check_confidence_alignment(classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if the confidence scores align with the predicted class.
    
    Args:
        classification: The classification to check
        
    Returns:
        dict: The updated classification with aligned scores
    """
    # Only relevant for service businesses
    if not classification.get("is_service_business", False):
        return classification
        
    if "confidence_scores" not in classification or "predicted_class" not in classification:
        return classification
        
    confidence_scores = classification["confidence_scores"]
    predicted_class = classification["predicted_class"]
    
    # Make sure predicted class has the highest score
    if predicted_class in confidence_scores:
        highest_score = max(confidence_scores.items(), key=lambda x: x[1] if x[0] != "Internal IT Department" else 0)[0]
        
        if highest_score != predicted_class:
            logger.warning(f"Predicted class {predicted_class} doesn't match highest confidence {highest_score}, adjusting")
            # Boost predicted class above the highest
            confidence_scores[predicted_class] = confidence_scores[highest_score] + 5
            
    # Make sure scores are differentiated
    service_scores = {k: v for k, v in confidence_scores.items() if k != "Internal IT Department"}
    if len(set(service_scores.values())) <= 1 or max(service_scores.values()) - min(service_scores.values()) < 5:
        logger.warning("Confidence scores not differentiated enough, adjusting")
        
        # Set up differentiated scores
        if predicted_class == "Managed Service Provider":
            confidence_scores.update({
                "Managed Service Provider": 80,
                "Integrator - Commercial A/V": 15,
                "Integrator - Residential A/V": 5
            })
        elif predicted_class == "Integrator - Commercial A/V":
            confidence_scores.update({
                "Integrator - Commercial A/V": 80,
                "Managed Service Provider": 15,
                "Integrator - Residential A/V": 5
            })
        else:  # Residential A/V
            confidence_scores.update({
                "Integrator - Residential A/V": 80,
                "Integrator - Commercial A/V": 15,
                "Managed Service Provider": 5
            })
            
    # Ensure Internal IT Department is set to 0 for service businesses
    confidence_scores["Internal IT Department"] = 0
            
    # Update classification
    classification["confidence_scores"] = confidence_scores
    return classification


def ensure_step_format(classification: Dict[str, Any], domain: str = None) -> Dict[str, Any]:
    """
    Ensure the explanation follows the step-by-step format.
    
    Args:
        classification: The classification dictionary
        domain: Optional domain name for context
        
    Returns:
        dict: The classification with properly formatted explanation
    """
    if "llm_explanation" not in classification:
        return classification
        
    explanation = classification["llm_explanation"]
    
    # Check if the explanation already has the STEP format
    if not any(f"STEP {i}" in explanation for i in range(1, 6)) and not any(f"STEP {i}:" in explanation for i in range(1, 6)):
        domain_name = domain or "This domain"
        predicted_class = classification.get("predicted_class", "Unknown")
        is_service = classification.get("is_service_business", False)
        
        # Create a structured explanation with STEP format
        new_explanation = f"Based on the website content, {domain_name} is classified as a {predicted_class}\n\n"
        new_explanation += f"STEP 1: The website content provides sufficient information to analyze and classify the business, so the processing status is successful\n\n"
        new_explanation += f"STEP 2: The domain is not parked, under construction, or for sale, so it is not a Parked Domain\n\n"
        
        if is_service:
            confidence = int(classification.get('max_confidence', 0.8) * 100)
            new_explanation += f"STEP 3: The company is a service business that provides services to other businesses\n\n"
            new_explanation += f"STEP 4: Based on the service offerings described, this company is classified as a {predicted_class} with {confidence}% confidence\n\n"
            new_explanation += f"STEP 5: Since this is classified as a service business, the internal IT potential is set to 0/100\n\n"
        else:
            it_potential = classification.get("internal_it_potential", 50)
            new_explanation += f"STEP 3: The company is NOT a service/management business that provides ongoing IT or A/V services to clients\n\n"
            new_explanation += f"STEP 4: Since this is not a service business, we classify it as Internal IT Department\n\n"
            new_explanation += f"STEP 5: As a non-service business, we assess its internal IT potential at {it_potential}/100\n\n"
            
        # Include the original explanation as a summary
        new_explanation += f"In summary: {explanation}"
        classification["llm_explanation"] = new_explanation
        
    return classification


def validate_result_consistency(result: Dict[str, Any], domain: str = None) -> Dict[str, Any]:
    """
    Validate and ensure consistency between predicted_class, confidence scores, and explanation.
    
    Args:
        result: The classification result
        domain: The domain
        
    Returns:
        dict: The validated and consistent result
    """
    if not result:
        return result
        
    # First, ensure predicted_class is never null
    if result.get("predicted_class") is None:
        # Extract a class from the explanation if possible
        explanation = result.get("explanation", "")
        
        # Look for existing company type in explanation
        if "managed service provider" in explanation.lower() or "msp" in explanation.lower():
            result["predicted_class"] = "Managed Service Provider"
        elif "commercial a/v" in explanation.lower() or "commercial integrator" in explanation.lower():
            result["predicted_class"] = "Integrator - Commercial A/V"
        elif "residential a/v" in explanation.lower() or "residential integrator" in explanation.lower():
            result["predicted_class"] = "Integrator - Residential A/V"
        elif "non-service business" in explanation.lower() or "not a service" in explanation.lower():
            result["predicted_class"] = "Internal IT Department"
        elif "vacation rental" in explanation.lower() or "travel" in explanation.lower():
            result["predicted_class"] = "Internal IT Department"
        elif "parked domain" in explanation.lower() or "website is parked" in explanation.lower():
            result["predicted_class"] = "Parked Domain"
        else:
            # Default to Unknown
            result["predicted_class"] = "Unknown"
            
        logger.warning(f"Fixed null predicted_class for {domain} to {result['predicted_class']}")
    
    # Check for cases where confidence score is very low for service businesses
    # Only do this for fresh classifications, not cached ones
    if "confidence_score" in result and result["confidence_score"] <= 15 and result.get("source") != "cached":
        explanation = result.get("explanation", "").lower()
        if "non-service business" in explanation or "not a service" in explanation:
            if result["predicted_class"] in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                # If explanation mentions non-service but class is a service type, fix it
                logger.warning(f"Correcting predicted_class from {result['predicted_class']} to Internal IT Department based on explanation")
                result["predicted_class"] = "Internal IT Department"
                
    # Ensure "Process Did Not Complete" has 0% confidence
    if result.get("predicted_class") == "Process Did Not Complete":
        result["confidence_score"] = 0
        if "confidence_scores" in result:
            result["confidence_scores"] = {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Internal IT Department": 0
            }
    
    # Make sure we have confidence_scores
    if "confidence_scores" not in result:
        result["confidence_scores"] = {}
        
    # For Internal IT Department or Non-Service Business, ensure Internal IT Department score is included
    if result.get("predicted_class") == "Internal IT Department" or result.get("predicted_class") == "Non-Service Business":
        # Convert Non-Service Business to Internal IT Department for consistency
        if result.get("predicted_class") == "Non-Service Business":
            result["predicted_class"] = "Internal IT Department"
            
        # Add Internal IT Department if not present
        if "Internal IT Department" not in result["confidence_scores"]:
            # Try to extract IT potential from explanation
            explanation = result.get("explanation", "")
            it_potential = 60  # Default
            
            # Look for internal IT potential in explanation
            it_match = re.search(r'internal IT.*?(\d+)[/\s]*100', explanation.lower())
            if it_match:
                try:
                    it_potential = int(it_match.group(1))
                except (ValueError, TypeError):
                    pass
            
            # Update confidence scores
            result["confidence_scores"]["Internal IT Department"] = it_potential
            result["confidence_scores"]["Managed Service Provider"] = 5
            result["confidence_scores"]["Integrator - Commercial A/V"] = 3
            result["confidence_scores"]["Integrator - Residential A/V"] = 2
            
            # Set overall confidence to 80% for non-service
            result["confidence_score"] = 80
    
    # For service businesses, ensure Internal IT Department is included with 0 score
    elif result.get("predicted_class") in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
        if "Internal IT Department" not in result["confidence_scores"]:
            result["confidence_scores"]["Internal IT Department"] = 0
    
    # For unknown or parked domains, ensure Internal IT Department is included with 0 score
    elif result.get("predicted_class") in ["Unknown", "Parked Domain"]:
        if "Internal IT Department" not in result["confidence_scores"]:
            result["confidence_scores"]["Internal IT Department"] = 0
    
    # Fix step numbering if it's off (e.g. starting at 6 instead of 1)
    explanation = result.get("explanation", "")
    fixed_steps = []
    step_pattern = re.compile(r'step\s*(\d+)[:\.]?\s*([^$]+?)(?=step\s*\d+[:\.]|$)', re.IGNORECASE | re.DOTALL)
    matches = step_pattern.findall(explanation)
    
    if matches and int(matches[0][0]) > 5:  # If steps start higher than 5
        # Renumber all steps to start from 1
        for i, (_, content) in enumerate(matches, 1):
            fixed_steps.append(f"STEP {i}: {content.strip()}")
            
        # Rebuild explanation with fixed step numbering
        if fixed_steps:
            result["explanation"] = "\n\n".join(fixed_steps)
    
    # Ensure explanation has step-by-step format if it's not a parked domain or process did not complete
    if (result.get("predicted_class") not in ["Parked Domain", "Process Did Not Complete", "Unknown"] 
        and "explanation" in result):
        explanation = result["explanation"]
        
        # Check if the explanation already has the STEP format
        if not any(f"STEP {i}" in explanation for i in range(1, 6)) and not any(f"STEP {i}:" in explanation for i in range(1, 6)):
            # If not already in step format and not numbered like "1:", "2:", etc.
            if not any(f"{i}:" in explanation for i in range(1, 6)):
                domain_name = domain or "This domain"
                predicted_class = result.get("predicted_class", "Unknown")
                is_service = predicted_class in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]
                
                # Create a structured explanation with STEP format
                new_explanation = f"Based on the website content, {domain_name} is classified as a {predicted_class}\n\n"
                new_explanation += f"STEP 1: The website content provides sufficient information to analyze and classify the business, so the processing status is successful\n\n"
                new_explanation += f"STEP 2: The domain is not parked, under construction, or for sale, so it is not a Parked Domain\n\n"
                
                if is_service:
                    confidence = result.get("confidence_score", 80)
                    new_explanation += f"STEP 3: The company is a service business that provides services to other businesses\n\n"
                    new_explanation += f"STEP 4: Based on the service offerings described, this company is classified as a {predicted_class} with {confidence}% confidence\n\n"
                    new_explanation += f"STEP 5: Since this is classified as a service business, the internal IT potential is set to 0/100\n\n"
                else:
                    # Try to extract internal IT potential from confidence scores
                    it_potential = 50  # Default
                    if "confidence_scores" in result and "Internal IT Department" in result["confidence_scores"]:
                        it_potential = result["confidence_scores"]["Internal IT Department"]
                    
                    new_explanation += f"STEP 3: The company is NOT a service/management business that provides ongoing IT or A/V services to clients\n\n"
                    new_explanation += f"STEP 4: Since this is not a service business, we classify it as Internal IT Department\n\n"
                    new_explanation += f"STEP 5: As a non-service business, we assess its internal IT potential at {it_potential}/100\n\n"
                    
                # Include the original explanation as a summary
                new_explanation += f"In summary: {explanation}"
                result["explanation"] = new_explanation
    
    # Ensure explanation is consistent with predicted_class
    if result.get("explanation") and "based on" in result["explanation"].lower():
        explanation = result["explanation"]
        # If explanation mentions company was "previously classified as a None"
        if "previously classified as a None" in explanation:
            # Fix this wording
            explanation = explanation.replace(
                f"previously classified as a None", 
                f"previously classified as a {result.get('predicted_class', 'company')}"
            )
            result["explanation"] = explanation
    
    # Handle any legacy "Corporate IT" keys that might still exist
    if "Corporate IT" in result.get("confidence_scores", {}):
        score = result["confidence_scores"].pop("Corporate IT")
        result["confidence_scores"]["Internal IT Department"] = score
        
    # Handle legacy "Non-Service Business" predicted class
    if result.get("predicted_class") == "Non-Service Business":
        result["predicted_class"] = "Internal IT Department"
    
    # Ensure company_description is present
    if "company_description" not in result and "explanation" in result:
        from domain_classifier.utils.text_processing import extract_company_description
        result["company_description"] = extract_company_description(
            "", result.get("explanation", ""), domain
        )
    
    return result
