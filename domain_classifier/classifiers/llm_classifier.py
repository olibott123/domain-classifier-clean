import requests
import logging
import json
import re
import os
from typing import Dict, Any, Optional

from domain_classifier.classifiers.prompt_builder import build_decision_tree_prompt, load_examples_from_knowledge_base
from domain_classifier.classifiers.decision_tree import (
    is_parked_domain,
    check_special_domain_cases,
    create_process_did_not_complete_result,
    create_parked_domain_result
)
from domain_classifier.utils.text_processing import (
    extract_json,
    clean_json_string,
    detect_minimal_content
)
from domain_classifier.classifiers.result_validator import validate_classification, check_confidence_alignment, ensure_step_format
from domain_classifier.classifiers.fallback_classifier import fallback_classification, parse_free_text

# Set up logging
logger = logging.getLogger(__name__)

class LLMClassifier:
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the LLM classifier with API key and model.
        
        Args:
            api_key: The API key for Claude API
            model: The Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. LLM classification will not be available.")
            
        self.model = model
        logger.info(f"Initialized LLM classifier with model: {model}")

    def classify(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Classify text content following the decision tree approach.
        
        Args:
            text_content: The text content to classify
            domain: Optional domain name for context
            
        Returns:
            dict: The classification results including predicted class and confidence scores
        """
        logger.info(f"Starting classification for domain: {domain or 'unknown'}")
        
        # STEP 1: Check if processing can complete
        if not text_content:
            logger.warning(f"No content provided for domain: {domain or 'unknown'}")
            return create_process_did_not_complete_result(domain)
        
        # Cache lowercase text for repeated use
        text_lower = text_content.lower()
        
        # STEP 2: Check if this is a parked/minimal domain
        if is_parked_domain(text_content):
            logger.info(f"Domain {domain or 'unknown'} is detected as a parked domain")
            return create_parked_domain_result(domain)
            
        is_minimal_content = detect_minimal_content(text_content)
        if is_minimal_content:
            logger.info(f"Domain {domain or 'unknown'} has minimal content")
            
        # Special case handling for specific domains
        if domain:
            domain_result = check_special_domain_cases(domain, text_content)
            if domain_result:
                return domain_result
            
        # STEP 3: Use the LLM for classification
        try:
            if not self.api_key:
                raise ValueError("No API key provided")
                
            # Load examples from knowledge base
            examples = load_examples_from_knowledge_base()
            
            # Log knowledge base usage
            total_examples = sum(len(examples[cat]) for cat in examples)
            logger.info(f"Loaded {total_examples} examples from knowledge base")
            
            # Build system prompt with the decision tree approach
            system_prompt = build_decision_tree_prompt(examples)
                
            # Limit the text content to avoid token limits
            max_chars = 12000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars]
                
            # Create the request to the Claude API
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": f"Domain name: {domain or 'unknown'}\n\nWebsite content to classify: {text_content}"}
                ],
                "max_tokens": 1500,
                "temperature": 0.1
            }
            
            # Make the request to Claude
            logger.info(f"Making request to Claude API for domain {domain or 'unknown'}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response_data = response.json()
            
            if "error" in response_data:
                logger.error(f"Error from Claude API: {response_data['error']}")
                raise Exception(f"Claude API error: {response_data['error']}")
                
            # Extract the response text
            if "content" in response_data and len(response_data["content"]) > 0:
                text_response = response_data["content"][0]["text"]
                logger.info(f"Received response from Claude API for {domain}")
            else:
                logger.error("No content in Claude response")
                raise Exception("No content in Claude response")
                
            # Try to extract JSON from the response
            json_str = extract_json(text_response)
            
            if json_str:
                try:
                    # Try to parse the JSON
                    parsed_json = json.loads(clean_json_string(json_str))
                    
                    # Validate and normalize the parsed JSON
                    parsed_json = validate_classification(parsed_json, domain)
                    
                    # Add detection method
                    parsed_json["detection_method"] = "llm_classification"
                    
                    # Set low_confidence flag based on highest score
                    max_confidence = parsed_json.get("max_confidence", 0.5)
                    if isinstance(max_confidence, str):
                        try:
                            max_confidence = float(max_confidence)
                        except (ValueError, TypeError):
                            max_confidence = 0
                            
                    parsed_json["low_confidence"] = max_confidence < 0.4 if parsed_json.get("is_service_business", False) else True
                    
                    logger.info(f"Successful LLM classification for {domain or 'unknown'}: {parsed_json['predicted_class']}")
                    
                    # Ensure the explanation has the step-by-step format
                    parsed_json = ensure_step_format(parsed_json, domain)
                    
                    # Add company description field for new modular design
                    if "company_description" not in parsed_json and "llm_explanation" in parsed_json:
                        from domain_classifier.utils.text_processing import extract_company_description
                        parsed_json["company_description"] = extract_company_description(
                            text_content,
                            parsed_json["llm_explanation"],
                            domain
                        )
                    
                    # Return the validated classification
                    return parsed_json
                    
                except Exception as e:
                    logger.error(f"Error parsing LLM response: {e}")
                    logger.error(f"JSON string: {json_str}")
            
            # If we get here, JSON parsing failed, try free text parsing
            logger.warning("Could not find JSON in LLM response, falling back to text parsing")
            parsed_result = parse_free_text(text_response, domain)
            if is_minimal_content:
                parsed_result["detection_method"] = "text_parsing_with_minimal_content"
            else:
                parsed_result["detection_method"] = "text_parsing"
            
            # Ensure the explanation has the step-by-step format
            parsed_result = ensure_step_format(parsed_result, domain)
            
            logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fall back to keyword-based classification
            result = fallback_classification(text_content, domain)
            if is_minimal_content:
                result["detection_method"] = result["detection_method"] + "_with_minimal_content"
                
            # Ensure the explanation has the step-by-step format
            result = ensure_step_format(result, domain)
            
            return result
