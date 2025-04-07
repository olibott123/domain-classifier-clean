import requests
import logging
import json
import time
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClassifier:
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", max_retries: int = 3):
        """
        Initialize the LLM Classifier.
        
        Args:
            api_key: Anthropic API key
            model: Model to use for classification
            max_retries: Maximum number of retries on API failure
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        self.class_options = [
            "Integrator - Commercial A/V",
            "Integrator - Residential A/V",
            "Managed Service Provider"
        ]
        
    def classify(self, content: str, domain: str = None) -> Dict[str, Any]:
        """
        Classify website content using Anthropic Claude.
        
        Args:
            content: Website content to classify
            domain: Domain name (optional)
            
        Returns:
            Dictionary with classification results
        """
        # Truncate content if it's too long (Claude has token limits)
        content_sample = content[:8000] if len(content) > 8000 else content
        
        domain_info = f" from {domain}" if domain else ""
        
        prompt = f"""
        You are a specialized business classifier that categorizes technology companies.
        
        Your task is to analyze website content{domain_info} and classify it into exactly ONE of these categories:
        - Integrator - Commercial A/V: Companies that primarily install and integrate audiovisual systems for commercial settings like offices, conference rooms, hospitals, etc.
        - Integrator - Residential A/V: Companies that primarily install home theater systems, whole-home audio, and smart home technology for residential customers.
        - Managed Service Provider (MSP): Companies that provide IT services, network management, cloud services, and technical support to businesses.
        
        Based on the website content, select the SINGLE most appropriate classification.
        
        IMPORTANT INSTRUCTIONS:
        1. If a company seems to serve both commercial and residential AV customers, classify based on their primary focus.
        2. MSPs focus on IT services, network management, and technical support - not primarily on audiovisual integration.
        3. Provide your confidence level (high/medium/low) based on how clear the evidence is.
        4. Include a brief explanation of your reasoning with specific evidence from the text.
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        Classification: [category name]
        Confidence: [high/medium/low]
        Explanation: [your explanation with specific evidence]
        
        Website content{domain_info}:
        {content_sample}
        """
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_anthropic_api(prompt)
                result = self._parse_response(response)
                return result
            except Exception as e:
                logger.error(f"Error classifying with LLM (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Return a fallback classification if all retries fail
                    return {
                        "predicted_class": "Unknown",
                        "confidence_scores": {option: 0.0 for option in self.class_options},
                        "max_confidence": 0.0,
                        "low_confidence": True,
                        "detection_method": "llm_classification_failed",
                        "explanation": f"LLM classification failed after {self.max_retries} attempts: {e}"
                    }
    
    def detect_internal_it_department(self, content: str, domain: str = None) -> Dict[str, Any]:
        """
        Special detector for internal IT departments at companies
        whose primary business is not IT services.
        
        Args:
            content: Website content to analyze
            domain: Domain name (optional)
            
        Returns:
            Dictionary with analysis results
        """
        # Truncate content if it's too long
        content_sample = content[:5000] if len(content) > 5000 else content
        
        domain_info = f" from {domain}" if domain else ""
        
        prompt = f"""
        Analyze the following website content and determine if it represents an INTERNAL IT DEPARTMENT 
        of a larger company (not an IT service provider or MSP).
        
        Internal IT departments:
        - Support only their own company (not external clients)
        - Are part of a larger organization whose main business is NOT IT services
        - Focus on supporting internal users, systems, and infrastructure
        - Are cost centers rather than profit centers
        - Don't market IT services to external customers
        
        Respond with:
        1. "Yes" or "No" - is this an internal IT department?
        2. Confidence: high/medium/low
        3. Brief explanation of your reasoning with specific evidence
        
        Website Content{domain_info}:
        {content_sample}
        """
        
        try:
            response = self._call_anthropic_api(prompt)
            answer_text = response["content"][0]["text"].strip()
            
            # Parse the response
            is_internal_it = "yes" in answer_text.lower().split("\n")[0].lower()
            
            # Extract confidence level
            confidence_text = "low"
            confidence_value = 0.5
            for line in answer_text.split("\n"):
                if "confidence:" in line.lower():
                    if "high" in line.lower():
                        confidence_text = "high"
                        confidence_value = 0.9
                    elif "medium" in line.lower():
                        confidence_text = "medium" 
                        confidence_value = 0.7
                    else:
                        confidence_text = "low"
                        confidence_value = 0.5
                    break
                    
            # Extract explanation
            explanation = ""
            explanation_started = False
            for line in answer_text.split("\n"):
                if explanation_started:
                    explanation += line + " "
                elif "explanation:" in line.lower():
                    explanation = line.replace("Explanation:", "").strip() + " "
                    explanation_started = True
            
            result = {
                "is_internal_it": is_internal_it,
                "confidence": confidence_value,
                "confidence_text": confidence_text,
                "explanation": explanation.strip()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting internal IT department: {e}")
            return {
                "is_internal_it": False,
                "confidence": 0.0,
                "confidence_text": "error",
                "explanation": f"Error analyzing: {str(e)}"
            }
    
    def _call_anthropic_api(self, prompt: str) -> Dict[str, Any]:
        """Call the Anthropic API with the given prompt"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            response.raise_for_status()
            
        return response.json()
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM response into a structured classification result"""
        try:
            # Extract text from the response
            answer = response["content"][0]["text"].strip()
            
            # Parse the formatted response
            lines = answer.split('\n')
            classification = None
            confidence_text = None
            explanation = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Classification:"):
                    classification = line.replace("Classification:", "").strip()
                elif line.startswith("Confidence:"):
                    confidence_text = line.replace("Confidence:", "").strip().lower()
                elif line.startswith("Explanation:"):
                    explanation = line.replace("Explanation:", "").strip()
                elif explanation:  # Append to explanation if we've already started it
                    explanation += " " + line
            
            # Validate classification
            if classification not in self.class_options:
                closest_match = self._find_closest_match(classification, self.class_options)
                logger.warning(f"LLM returned invalid class '{classification}', using closest match: '{closest_match}'")
                classification = closest_match
            
            # Convert confidence text to numeric
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
            confidence_score = confidence_map.get(confidence_text, 0.7)  # Default to medium if unrecognized
            
            # Build confidence scores for all classes
            # The chosen class gets the confidence score, others get proportions of remaining confidence
            total_remaining = 1.0 - confidence_score
            other_classes = [cls for cls in self.class_options if cls != classification]
            other_class_scores = {cls: total_remaining / len(other_classes) for cls in other_classes}
            
            confidence_scores = {**{classification: confidence_score}, **other_class_scores}
            
            return {
                "predicted_class": classification,
                "confidence_scores": confidence_scores,
                "max_confidence": confidence_score,
                "low_confidence": confidence_score < 0.7,
                "detection_method": "llm_classification",
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"Failed to parse LLM response: {e}")
    
    def _find_closest_match(self, text: str, options: List[str]) -> str:
        """Find the closest matching classification option"""
        text = text.lower()
        
        # First check for keyword matches
        if "commercial" in text and "a/v" in text or "av" in text:
            return "Integrator - Commercial A/V"
        elif "residential" in text and "a/v" in text or "av" in text:
            return "Integrator - Residential A/V"
        elif "msp" in text or "managed service" in text or "it service" in text:
            return "Managed Service Provider"
        
        # Default to the first option if no match
        return options[0]
