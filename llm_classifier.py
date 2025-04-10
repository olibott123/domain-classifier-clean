import requests
import logging
import json
import time
from typing import Dict, Any, List, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClassifier:
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        """Initialize the LLM classifier.
        
        Args:
            api_key: Anthropic API key
            model: Model to use for classification
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Define the categories and their key characteristics
        self.categories = {
            "Integrator - Commercial A/V": [
                "Focuses on audiovisual solutions for businesses and organizations",
                "Installs conference room equipment, digital signage, and presentation systems",
                "Clients are primarily corporate offices, educational institutions, government facilities",
                "Often handles enterprise-level projects with multiple rooms or buildings",
                "May mention certifications like CTS, Crestron, Extron, or Biamp"
            ],
            "Integrator - Residential A/V": [
                "Specializes in home theater and audio systems for private residences",
                "Provides home automation, smart home, and entertainment solutions",
                "Clients are homeowners and residential properties",
                "Focuses on living spaces, dedicated theater rooms, and whole-home audio",
                "May mention brands like Control4, Savant, Sonos, or mention luxury home experiences"
            ],
            "Managed Service Provider": [
                "Offers IT services, network management, and technical support",
                "Provides cloud services, cybersecurity, and help desk support",
                "Focuses on maintaining and optimizing IT infrastructure",
                "May mention server management, network monitoring, or data backup",
                "Often discusses SLAs, uptime guarantees, or managed security services"
            ]
        }
    
    def classify(self, text: str, domain: str = None) -> Dict[str, Any]:
        """Classify website text using Claude.
        
        Args:
            text: The website content to classify
            domain: Optional domain name for context
            
        Returns:
            Dict containing classification results
        """
        try:
            # Prepare the input for Claude
            domain_context = f" for the website {domain}" if domain else ""
            
            # Generate a prompt that asks for detailed reasoning
            system_prompt = f"""You are an expert at classifying companies based on their website content. Your task is to determine whether a company is a:

1. Commercial A/V Integrator - Companies that install audiovisual systems for businesses (conference rooms, offices, etc.)
2. Residential A/V Integrator - Companies that install home theaters, smart home systems, etc. for homeowners
3. Managed Service Provider (MSP) - IT service companies that provide technical support, network management, etc.

For each category, here are the key characteristics:

Commercial A/V Integrator:
- {chr(10).join(self.categories["Integrator - Commercial A/V"])}

Residential A/V Integrator:
- {chr(10).join(self.categories["Integrator - Residential A/V"])}

Managed Service Provider:
- {chr(10).join(self.categories["Managed Service Provider"])}

Analyze the provided website content carefully.
First, extract important clues about the company's business focus.
Then, determine which category best matches these clues.
Provide confidence scores that reflect how strongly the evidence supports each category.
Most importantly, provide detailed reasoning that explains WHY you believe this classification is correct.
"""

            user_prompt = f"Based on the following website content{domain_context}, classify the company and explain your reasoning in detail. Focus on specific evidence from the text that supports your conclusion:\n\n{text[:15000]}"

            response_format = {
                "predicted_class": "The most likely category (one of: 'Integrator - Commercial A/V', 'Integrator - Residential A/V', 'Managed Service Provider')",
                "confidence_scores": {
                    "Integrator - Commercial A/V": "float between 0 and 1",
                    "Integrator - Residential A/V": "float between 0 and 1",
                    "Managed Service Provider": "float between 0 and 1"
                },
                "llm_explanation": "A detailed explanation of why this classification was chosen, citing specific evidence from the text"
            }

            request_data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.0
            }
            
            # Make the API request with retries
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=request_data,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result.get("content", [])
                        
                        if content and len(content) > 0:
                            text_content = content[0].get("text", "")
                            
                            # Extract the JSON portion from the text response
                            # Look for structure with predicted_class and confidence_scores
                            try:
                                # Try to find JSON-like structure in the response
                                import re
                                json_str = re.search(r'({.*"predicted_class".*})', text_content, re.DOTALL)
                                
                                if not json_str:
                                    # Try to parse free-form text
                                    parsed_result = self._parse_free_text(text_content)
                                    logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
                                    return parsed_result
                                
                                parsed_json = json.loads(json_str.group(1))
                                
                                # Ensure all required fields are present
                                if "predicted_class" not in parsed_json:
                                    raise ValueError("Missing predicted_class in response")
                                
                                if "confidence_scores" not in parsed_json:
                                    # Generate confidence scores from text
                                    parsed_json["confidence_scores"] = self._extract_confidence_scores(text_content)
                                
                                # Ensure llm_explanation is included
                                if "llm_explanation" not in parsed_json:
                                    # Extract reasoning from the full text
                                    parsed_json["llm_explanation"] = self._extract_explanation(text_content)
                                
                                # Calculate max confidence
                                confidence_scores = parsed_json.get("confidence_scores", {})
                                if confidence_scores:
                                    parsed_json["max_confidence"] = max(confidence_scores.values())
                                else:
                                    parsed_json["max_confidence"] = 0.7  # Default confidence
                                
                                return parsed_json
                            except Exception as e:
                                logger.error(f"Error parsing LLM response: {e}")
                                # Fall back to manual parsing
                                return self._parse_free_text(text_content)
                        else:
                            raise ValueError("Empty response from Claude API")
                    elif response.status_code == 429:  # Rate limited
                        logger.warning(f"Rate limited by Claude API. Attempt {attempt+1}/{max_retries}")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        raise ValueError(f"API error: {response.status_code} - {response.text}")
                except Exception as e:
                    logger.error(f"Request error on attempt {attempt+1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
            
            raise ValueError("Failed to get response after maximum retries")
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Return a fallback classification
            return {
                "predicted_class": "Unknown",
                "confidence_scores": {
                    "Integrator - Commercial A/V": 0.0,
                    "Integrator - Residential A/V": 0.0,
                    "Managed Service Provider": 0.0
                },
                "max_confidence": 0.0,
                "llm_explanation": f"Classification failed due to an error: {str(e)}",
                "detection_method": "llm_classification_failed"
            }
    
    def _parse_free_text(self, text: str) -> Dict[str, Any]:
        """Parse classification from free-form text response.
        
        Args:
            text: The text response from the LLM
            
        Returns:
            Dict containing parsed classification results
        """
        result = {
            "confidence_scores": {
                "Integrator - Commercial A/V": 0.0,
                "Integrator - Residential A/V": 0.0,
                "Managed Service Provider": 0.0
            },
            "llm_explanation": ""
        }
        
        # Try to identify the predicted class
        if "commercial a/v" in text.lower() or "commercial av" in text.lower():
            result["predicted_class"] = "Integrator - Commercial A/V"
        elif "residential a/v" in text.lower() or "residential av" in text.lower() or "home theater" in text.lower():
            result["predicted_class"] = "Integrator - Residential A/V"
        elif "managed service" in text.lower() or "it service" in text.lower() or "msp" in text.lower():
            result["predicted_class"] = "Managed Service Provider"
        else:
            result["predicted_class"] = "Unknown"
        
        # Extract confidence scores
        result["confidence_scores"] = self._extract_confidence_scores(text)
        
        # Extract explanation
        result["llm_explanation"] = self._extract_explanation(text)
        
        # Calculate max confidence
        if result["confidence_scores"]:
            result["max_confidence"] = max(result["confidence_scores"].values())
        else:
            result["max_confidence"] = 0.7  # Default confidence
        
        return result
    
    def _extract_confidence_scores(self, text: str) -> Dict[str, float]:
        """Extract confidence scores from text response.
        
        Args:
            text: The text response from the LLM
            
        Returns:
            Dict containing confidence scores for each category
        """
        scores = {
            "Integrator - Commercial A/V": 0.0,
            "Integrator - Residential A/V": 0.0,
            "Managed Service Provider": 0.0
        }
        
        # Look for explicit confidence values
        import re
        
        # Check for Commercial A/V confidence
        commercial_match = re.search(r"commercial\s*a\/v.*?(\d+(?:\.\d+)?)%", text.lower())
        if not commercial_match:
            commercial_match = re.search(r"commercial\s*a\/v.*?confidence.*?(\d+(?:\.\d+)?)", text.lower())
        if commercial_match:
            try:
                scores["Integrator - Commercial A/V"] = float(commercial_match.group(1)) / 100
            except:
                pass
        
        # Check for Residential A/V confidence
        residential_match = re.search(r"residential\s*a\/v.*?(\d+(?:\.\d+)?)%", text.lower())
        if not residential_match:
            residential_match = re.search(r"residential\s*a\/v.*?confidence.*?(\d+(?:\.\d+)?)", text.lower())
        if residential_match:
            try:
                scores["Integrator - Residential A/V"] = float(residential_match.group(1)) / 100
            except:
                pass
        
        # Check for MSP confidence
        msp_match = re.search(r"managed service provider.*?(\d+(?:\.\d+)?)%", text.lower())
        if not msp_match:
            msp_match = re.search(r"msp.*?confidence.*?(\d+(?:\.\d+)?)", text.lower())
        if msp_match:
            try:
                scores["Managed Service Provider"] = float(msp_match.group(1)) / 100
            except:
                pass
        
        # If no scores were extracted, assign based on predicted class
        if all(score == 0.0 for score in scores.values()):
            if "commercial a/v" in text.lower() or "commercial av" in text.lower():
                scores["Integrator - Commercial A/V"] = 0.8
                scores["Integrator - Residential A/V"] = 0.1
                scores["Managed Service Provider"] = 0.1
            elif "residential a/v" in text.lower() or "residential av" in text.lower():
                scores["Integrator - Commercial A/V"] = 0.1
                scores["Integrator - Residential A/V"] = 0.8
                scores["Managed Service Provider"] = 0.1
            elif "managed service" in text.lower() or "msp" in text.lower():
                scores["Integrator - Commercial A/V"] = 0.1
                scores["Integrator - Residential A/V"] = 0.1
                scores["Managed Service Provider"] = 0.8
            else:
                scores["Integrator - Commercial A/V"] = 0.33
                scores["Integrator - Residential A/V"] = 0.33
                scores["Managed Service Provider"] = 0.34
        
        return scores
    
    def _extract_explanation(self, text: str) -> str:
        """Extract the explanation from the text response.
        
        Args:
            text: The text response from the LLM
            
        Returns:
            The extracted explanation
        """
        # Look for the explanation section
        explanation = ""
        
        # Check for specific markers
        markers = [
            "reasoning:", "explanation:", "analysis:", "evidence:", 
            "rationale:", "justification:", "conclusion:"
        ]
        
        for marker in markers:
            if marker in text.lower():
                parts = text.lower().split(marker)
                if len(parts) > 1:
                    explanation = parts[1].strip()
                    break
        
        # If no marker found, use the entire text
        if not explanation:
            explanation = text
        
        # Clean up the explanation
        # Remove any JSON-like formatting
        explanation = re.sub(r'{.*}', '', explanation, flags=re.DOTALL)
        
        # Limit length
        if len(explanation) > 1000:
            explanation = explanation[:997] + "..."
        
        return explanation
