import requests
import logging
import json
import time
import re
from typing import Dict, Any, List, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClassifier:
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the LLM Classifier with the Anthropic API key.
        
        Args:
            api_key: The Anthropic API key (defaults to environment variable)
            model: The model to use for classification
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and ANTHROPIC_API_KEY not set in environment")
        
        self.model = model
        logger.info(f"Initialized LLM classifier with model: {model}")
        
        # Define keywords and indicators
        self.msp_indicators = [
            "managed service provider", "msp", "it services", "it support", "managed it", 
            "cloud services", "network management", "it consulting", "cybersecurity",
            "helpdesk", "service desk", "remote monitoring", "rmm", "psa", "it outsourcing",
            "business continuity", "disaster recovery", "server management", "endpoint management",
            "patch management", "network security", "backup solutions", "it infrastructure",
            "managed security", "remote support", "desktop support", "computer repair",
            "technical support", "break/fix", "data center", "it management", "it procurement",
            "iaas", "saas", "paas"
        ]
        
        self.commercial_av_indicators = [
            "commercial av", "integrator", "av integrator", "commercial integration", 
            "conference room", "meeting room", "boardroom", "digital signage", "presentation systems",
            "sound systems", "commercial audio", "corporate av", "commercial video", "video walls",
            "distributed audio", "commercial displays", "commercial projectors", "large venue",
            "command center", "control room", "huddle room", "room scheduling", "commercial automation",
            "UC", "unified communications", "video conferencing", "collaboration technology",
            "enterprise", "corporate", "business", "commercial", "B2B", "building management"
        ]
        
        self.residential_av_indicators = [
            "home automation", "smart home", "residential", "home theater", "multi-room audio",
            "whole house audio", "custom installation", "home cinema", "residential integration",
            "home control", "smart lighting", "residential networking", "distributed video",
            "home entertainment", "residential av", "home av", "home technology", "consumer",
            "homeowner", "residential clients", "home security", "home network", "media room", 
            "living room", "whole-home", "custom home", "luxury home", "smart home control",
            "family", "home", "house", "apartment", "condo", "residential",
            "consumer", "smart tv", "domestic", "household"
        ]

    def clean_json_string(self, json_str: str) -> str:
        """Clean a JSON string by removing control characters and fixing common issues."""
        # Replace common control characters
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        # Fix unescaped quotes within strings
        cleaned = re.sub(r'([^\\])"([^{}\[\],:"\\])', r'\1\"\2', cleaned)
        return cleaned

    def detect_parked_domain(self, content: str) -> bool:
        """
        Detect if a domain appears to be parked or has minimal content.
        
        Args:
            content: The website content
            
        Returns:
            bool: True if the domain appears to be parked/minimal
        """
        if not content or len(content.strip()) < 100:
            logger.info(f"Domain content is very short: {len(content) if content else 0} characters")
            return True
            
        # Check for common parking phrases
        parking_phrases = [
            "domain is for sale", "buy this domain", "purchasing this domain", 
            "domain may be for sale", "parked domain", "coming soon", "under construction",
            "website coming soon", "site coming soon", "this website is for sale",
            "godaddy", "domain registration", "domain registrar", "hostinger", "namecheap"
        ]
        
        # Count words in content
        words = re.findall(r'\b\w+\b', content.lower())
        unique_words = set(words)
        
        if len(words) < 50:
            logger.info(f"Domain has only {len(words)} words, likely minimal content")
            return True
            
        if len(unique_words) < 30:
            logger.info(f"Domain has only {len(unique_words)} unique words, likely minimal content")
            return True
            
        # Check if content contains parking phrases
        for phrase in parking_phrases:
            if phrase in content.lower():
                logger.info(f"Domain contains parking phrase: '{phrase}'")
                return True
                
        # Check if basic business terms are missing
        basic_business_terms = ["contact", "about", "service", "product", "company", "business", "client"]
        found_terms = sum(1 for term in basic_business_terms if term in content.lower())
        
        if found_terms <= 1:
            logger.info(f"Domain lacks basic business terms, found only {found_terms}/7 terms")
            return True
            
        return False

    def classify(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Classify text content to determine company type.
        
        Args:
            text_content: The text content to classify
            domain: Optional domain name for context
            
        Returns:
            dict: The classification results including predicted class and confidence scores
        """
        # Check if domain appears to be parked or has minimal content
        if self.detect_parked_domain(text_content):
            logger.warning(f"Domain {domain or 'unknown'} appears to be parked or has minimal content")
            return {
                "predicted_class": "Integrator - Commercial A/V",  # Default to avoid breaking changes
                "confidence_scores": {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5
                },
                "llm_explanation": "This domain appears to be parked or has minimal content. Unable to determine the company type with confidence. The classification is inconclusive due to insufficient information.",
                "detection_method": "minimal_content_detection",
                "low_confidence": True
            }
            
        try:
            # Define system prompt with classification task
            system_prompt = f"""You are an expert business analyst who specializes in categorizing technology companies.
Your task is to analyze the text from a company's website and classify it into ONE of these categories:
1. Managed Service Provider (MSP)
2. Integrator - Commercial A/V
3. Integrator - Residential A/V

First look at the text carefully, noting any services, products, clients, or terminology that indicates the company's focus.

MSPs typically offer IT services, network management, cybersecurity, cloud services, and technical support to businesses.

Commercial A/V Integrators focus on audiovisual technology for businesses, such as conference rooms, digital signage, meeting spaces, and enterprise-level communication systems.

Residential A/V Integrators specialize in home theater, smart home technology, residential automation, and audio/video systems for homes.

You MUST provide your analysis in JSON format with the following structure:
{{
  "predicted_class": "The most likely category (one of: 'Integrator - Commercial A/V', 'Integrator - Residential A/V', 'Managed Service Provider')",
  "confidence_scores": {{
    "Integrator - Commercial A/V": Integer from 1-100,
    "Integrator - Residential A/V": Integer from 1-100,
    "Managed Service Provider": Integer from 1-100
  }},
  "llm_explanation": "A detailed explanation of why this classification was chosen, citing specific evidence from the text"
}}

IMPORTANT GUIDELINES:
- You must provide DIFFERENT confidence scores for each category - they should NOT all be the same value.
- The scores should reflect how confident you are that the company belongs to each category, with higher numbers indicating higher confidence.
- Your llm_explanation field MUST be detailed (at least 3-4 sentences) and explain the reasoning behind your classification, citing specific evidence found in the text.
- If there isn't enough information in the text, assign low confidence scores (below 30) and note this in your explanation.
- Ensure your JSON is properly formatted with no trailing commas.
"""

            # Make request to Anthropic API
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "x-api-key": self.api_key
            }
            
            data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Here's the text from a company's website{' (' + domain + ')' if domain else ''}: \n\n{text_content}"
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.1
            }
            
            # Add timeout to avoid hanging requests
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response_data = response.json()
            
            if "error" in response_data:
                logger.error(f"API Error: {response_data['error']}")
                return self._fallback_classification(text_content)
                
            # Extract the text response from the API
            text_response = response_data["content"][0]["text"]
            
            # Try to find JSON-like structure in the response
            json_match = re.search(r'({.*"predicted_class".*})', text_response, re.DOTALL)
            
            if json_match:
                try:
                    # Clean the JSON string to handle potential issues
                    cleaned_json = self.clean_json_string(json_match.group(1))
                    parsed_json = json.loads(cleaned_json)
                    
                    # Validate and process confidence scores
                    if "confidence_scores" in parsed_json:
                        # Convert confidence scores to integers between 1-100
                        for category in parsed_json["confidence_scores"]:
                            score_value = parsed_json["confidence_scores"][category]
                            
                            # Handle potential strings that look like numbers
                            if isinstance(score_value, str) and score_value.isdigit():
                                score_value = int(score_value)
                            
                            # Ensure score is an integer between 1-100
                            if isinstance(score_value, (int, float)):
                                # If score is between 0-1, convert to 1-100 scale
                                if 0 <= score_value <= 1:
                                    score_value = int(score_value * 100)
                                
                                # Ensure it's within bounds
                                score_value = max(1, min(100, int(score_value)))
                                parsed_json["confidence_scores"][category] = score_value
                            else:
                                # Default if not a valid number
                                parsed_json["confidence_scores"][category] = 10
                    
                    # Check if all scores are identical and fix if needed
                    scores = list(parsed_json["confidence_scores"].values())
                    if len(set(scores)) == 1:
                        logger.warning("All confidence scores are the same, regenerating differentiated scores")
                        # Find predicted class and make it have the highest score
                        predicted_class = parsed_json["predicted_class"]
                        base_score = max(scores[0] - 20, 5)  # Reduce by 20 but keep above 5
                        
                        # Set different scores for each class
                        for category in parsed_json["confidence_scores"]:
                            if category == predicted_class:
                                parsed_json["confidence_scores"][category] = min(base_score + 30, 100)
                            elif category == list(parsed_json["confidence_scores"].keys())[0]:
                                parsed_json["confidence_scores"][category] = min(base_score + 15, 95)
                            else:
                                parsed_json["confidence_scores"][category] = base_score
                    
                    # Ensure explanation is substantial
                    if "llm_explanation" not in parsed_json or not parsed_json["llm_explanation"]:
                        parsed_json["llm_explanation"] = self._extract_explanation(text_response)
                    
                    # If explanation is generic, try to extract a better one
                    generic_explanations = [
                        "Classification based on analysis of website content",
                        "Based on the available information",
                        "Based on the content provided"
                    ]
                    
                    explanation = parsed_json["llm_explanation"]
                    if any(generic in explanation for generic in generic_explanations) or len(explanation) < 50:
                        better_explanation = self._extract_explanation(text_response)
                        if len(better_explanation) > len(explanation):
                            parsed_json["llm_explanation"] = better_explanation
                    
                    parsed_json["detection_method"] = "llm"
                    logger.info(f"Classified as {parsed_json['predicted_class']} with LLM")
                    
                    # Add low_confidence flag based on highest confidence score
                    max_confidence = max(parsed_json["confidence_scores"].values())
                    parsed_json["low_confidence"] = max_confidence < 40
                    
                    # Check and fix if any confidence scores are outside 1-100 range
                    for category in parsed_json["confidence_scores"]:
                        score = parsed_json["confidence_scores"][category]
                        if not isinstance(score, int) or score < 1 or score > 100:
                            # If score is between 0-1, convert to 1-100 scale
                            if isinstance(score, (int, float)) and 0 <= score <= 1:
                                score = int(score * 100)
                            else:
                                # Default for invalid scores
                                score = 10 if category != parsed_json["predicted_class"] else 30
                            
                            parsed_json["confidence_scores"][category] = score
                    
                    return parsed_json
                    
                except Exception as e:
                    logger.error(f"Error parsing LLM response: {e}")
                    # Fall back to text parsing
            
            # If we get here, either no JSON was found or parsing failed
            # Try to parse free-form text
            logger.warning("Could not find JSON in LLM response, falling back to text parsing")
            parsed_result = self._parse_free_text(text_response)
            logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._fallback_classification(text_content)

    def _extract_explanation(self, text_content: str) -> str:
        """
        Extract a meaningful explanation from the LLM response text.
        
        Args:
            text_content: The LLM response text
            
        Returns:
            str: The extracted explanation
        """
        # Try different patterns to extract explanation
        explanation_patterns = [
            r'"llm_explanation":\s*"([^"]+)"',
            r'"llm_explanation":\s*"(.+?)"(?=,|\s*})',
            r'llm_explanation["\']:\s*["\'](.+?)["\'](?=,|\s*})',
            r'(?:explanation|reasoning|because|evidence)(?:[:\s])(.+?)(?:\.|\n|$)',
            r'(?:the company|this appears to be|classification)(.+?)(?:\.|$)',
            r'(?:based on|according to)(.+?)(?:\.|$)'
        ]
        
        # Try to extract explanation using patterns
        for pattern in explanation_patterns:
            matches = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
            if matches:
                explanation = matches.group(1).strip()
                # Clean up any JSON artifacts
                explanation = re.sub(r'[\\"]', '', explanation)
                if len(explanation) > 50:
                    return explanation
        
        # Extract sentences that contain reasoning keywords
        reasoning_keywords = ["because", "evidence", "indicates", "suggests", "appears", "likely", "based on"]
        sentences = re.split(r'(?<=[.!?])\s+', text_content)
        reasoning_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in reasoning_keywords)]
        
        if reasoning_sentences:
            return " ".join(reasoning_sentences[:3])
            
        # Last resort: just return the first few sentences
        if sentences:
            return " ".join(sentences[:3])
            
        # Ultimate fallback
        return "Classification based on analysis of website content. Insufficient information available for detailed explanation."

    def _extract_numeric_confidence(self, text, default=20) -> int:
        """
        Extract a numeric confidence value from text.
        
        Args:
            text: The text to extract from
            default: Default value if no confidence found
            
        Returns:
            int: The confidence value (1-100)
        """
        # Try to extract a percentage
        percentage_match = re.search(r'(\d+)%', text)
        if percentage_match:
            value = int(percentage_match.group(1))
            return max(1, min(100, value))
            
        # Try to extract a confidence score mentioned explicitly
        confidence_patterns = [
            r'confidence(?:\s+score)?(?:\s+of)?(?:\s+is)?(?:\s*:)?\s*(\d+)',
            r'score(?:\s+of)?(?:\s*:)?\s*(\d+)'
        ]
        
        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, text, re.IGNORECASE)
            if confidence_match:
                value = int(confidence_match.group(1))
                return max(1, min(100, value))
                
        # Try to find a decimal between 0-1
        decimal_match = re.search(r'confidence(?:\s+score)?(?:\s+of)?(?:\s+is)?(?:\s*:)?\s*(0\.\d+)', text, re.IGNORECASE)
        if decimal_match:
            value = float(decimal_match.group(1))
            return max(1, min(100, int(value * 100)))
            
        # Return the default
        return default

    def _parse_free_text(self, text_content: str) -> Dict[str, Any]:
        """
        Parse free-form text to extract classification and confidence scores.
        
        Args:
            text_content: The text to parse
            
        Returns:
            dict: The parsed classification results
        """
        result = {
            "predicted_class": None,
            "confidence_scores": {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0
            },
            "llm_explanation": self._extract_explanation(text_content),
            "detection_method": "text_parsing",
            "low_confidence": True
        }
        
        # Check for insufficient information indicators
        insufficient_info = any(phrase in text_content.lower() for phrase in [
            "insufficient information", "not enough information", "limited information",
            "hard to determine", "difficult to classify", "cannot determine"
        ])
        
        if insufficient_info:
            # Default to commercial AV with very low confidence
            result["predicted_class"] = "Integrator - Commercial A/V"
            result["confidence_scores"]["Integrator - Commercial A/V"] = 15
            result["confidence_scores"]["Integrator - Residential A/V"] = 10
            result["confidence_scores"]["Managed Service Provider"] = 5
            return result
        
        # Count indicators for each category
        text_lower = text_content.lower()
        
        msp_score = sum(1 for indicator in self.msp_indicators if indicator in text_lower)
        commercial_score = sum(1 for indicator in self.commercial_av_indicators if indicator in text_lower)
        residential_score = sum(1 for indicator in self.residential_av_indicators if indicator in text_lower)
        
        # Check direct mentions of each category
        if "managed service provider" in text_lower or "msp" in text_lower:
            msp_score += 2
        if "commercial av" in text_lower or "commercial integrator" in text_lower:
            commercial_score += 2
        if "residential av" in text_lower or "home automation" in text_lower:
            residential_score += 2
            
        total_score = msp_score + commercial_score + residential_score
        
        # Calculate dynamic confidence scores (without normalization)
        if total_score > 0:
            # Base confidence calculations on indicator counts
            result["confidence_scores"]["Managed Service Provider"] = min(5 + (msp_score * 5), 95) if msp_score > 0 else 5
            result["confidence_scores"]["Integrator - Commercial A/V"] = min(5 + (commercial_score * 5), 95) if commercial_score > 0 else 5
            result["confidence_scores"]["Integrator - Residential A/V"] = min(5 + (residential_score * 5), 95) if residential_score > 0 else 5
            
            # Determine predicted class based on scores
            if msp_score > commercial_score and msp_score > residential_score:
                result["predicted_class"] = "Managed Service Provider"
            elif commercial_score > msp_score and commercial_score > residential_score:
                result["predicted_class"] = "Integrator - Commercial A/V"
            elif residential_score > msp_score and residential_score > commercial_score:
                result["predicted_class"] = "Integrator - Residential A/V"
            else:
                # In case of a tie, look for direct mentions of category names
                if "managed service provider" in text_lower or "msp" in text_lower:
                    result["predicted_class"] = "Managed Service Provider"
                elif "commercial" in text_lower and "audio visual" in text_lower:
                    result["predicted_class"] = "Integrator - Commercial A/V"
                elif "residential" in text_lower and "audio visual" in text_lower:
                    result["predicted_class"] = "Integrator - Residential A/V"
                else:
                    # Fall back to the category with most explicit mentions
                    explicit_mentions = {
                        "Managed Service Provider": text_lower.count("managed service") + text_lower.count("msp") + text_lower.count("it service"),
                        "Integrator - Commercial A/V": text_lower.count("commercial") + text_lower.count("conference") + text_lower.count("corporate"),
                        "Integrator - Residential A/V": text_lower.count("residential") + text_lower.count("home") + text_lower.count("theater")
                    }
                    
                    max_mentions = max(explicit_mentions.values())
                    if max_mentions > 0:
                        for category, mentions in explicit_mentions.items():
                            if mentions == max_mentions:
                                result["predicted_class"] = category
                                break
                    else:
                        # Ultimate fallback - commercial AV is the middle ground
                        result["predicted_class"] = "Integrator - Commercial A/V"
        else:
            # If no indicators found, default to commercial AV with low confidence
            result["predicted_class"] = "Integrator - Commercial A/V"
            result["confidence_scores"]["Integrator - Commercial A/V"] = 20
            result["confidence_scores"]["Integrator - Residential A/V"] = 15
            result["confidence_scores"]["Managed Service Provider"] = 10
        
        # Ensure the predicted class always has the highest confidence score
        highest_category = max(result["confidence_scores"], key=result["confidence_scores"].get)
        if highest_category != result["predicted_class"]:
            # Swap values to ensure predicted class has highest confidence
            temp = result["confidence_scores"][highest_category]
            result["confidence_scores"][highest_category] = result["confidence_scores"][result["predicted_class"]]
            result["confidence_scores"][result["predicted_class"]] = temp
            
        # Set low confidence flag based on highest score
        max_confidence = max(result["confidence_scores"].values())
        result["low_confidence"] = max_confidence < 40
            
        return result

    def _fallback_classification(self, text_content: str) -> Dict[str, Any]:
        """
        Fallback classification method when API calls or parsing fails.
        
        Args:
            text_content: The text to classify
            
        Returns:
            dict: A default classification with low confidence
        """
        # Count indicators for each category to provide some basis for the fallback
        text_lower = text_content.lower()
        
        msp_count = sum(1 for indicator in self.msp_indicators if indicator in text_lower)
        commercial_count = sum(1 for indicator in self.commercial_av_indicators if indicator in text_lower)
        residential_count = sum(1 for indicator in self.residential_av_indicators if indicator in text_lower)
        
        total_count = msp_count + commercial_count + residential_count
        
        if total_count > 0:
            # If we have some indicators, use them to determine the most likely category
            if msp_count > commercial_count and msp_count > residential_count:
                predicted_class = "Managed Service Provider"
                confidence = {
                    "Managed Service Provider": min(5 + (msp_count * 3), 40),
                    "Integrator - Commercial A/V": min(5 + (commercial_count * 2), 30),
                    "Integrator - Residential A/V": min(5 + (residential_count * 2), 30)
                }
            elif commercial_count > msp_count and commercial_count > residential_count:
                predicted_class = "Integrator - Commercial A/V"
                confidence = {
                    "Integrator - Commercial A/V": min(5 + (commercial_count * 3), 40),
                    "Managed Service Provider": min(5 + (msp_count * 2), 30),
                    "Integrator - Residential A/V": min(5 + (residential_count * 2), 30)
                }
            elif residential_count > msp_count and residential_count > commercial_count:
                predicted_class = "Integrator - Residential A/V"
                confidence = {
                    "Integrator - Residential A/V": min(5 + (residential_count * 3), 40),
                    "Managed Service Provider": min(5 + (msp_count * 2), 30),
                    "Integrator - Commercial A/V": min(5 + (commercial_count * 2), 30)
                }
            else:
                # In case of a tie, default to commercial AV
                predicted_class = "Integrator - Commercial A/V"
                confidence = {
                    "Integrator - Commercial A/V": 30,
                    "Managed Service Provider": 25,
                    "Integrator - Residential A/V": 20
                }
        else:
            # If no indicators found, provide a very low confidence default
            predicted_class = "Integrator - Commercial A/V"
            confidence = {
                "Integrator - Commercial A/V": 20,
                "Managed Service Provider": 15,
                "Integrator - Residential A/V": 10
            }
            
        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence,
            "llm_explanation": "Classification based on keyword analysis. Limited information available for a confident classification.",
            "detection_method": "fallback",
            "low_confidence": True
        }
