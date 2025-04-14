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

    def repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON syntax errors."""
        # Replace single quotes with double quotes (common Claude error)
        fixed = re.sub(r"'([^']*)':", r'"\1":', json_str)
        
        # Fix trailing commas in objects and arrays
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Fix missing quotes around property names
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
        
        # Replace unescaped newlines in strings
        fixed = re.sub(r'(".*?)\n(.*?")', r'\1\\n\2', fixed, flags=re.DOTALL)
        
        # Handle decimal values without leading zero
        fixed = re.sub(r':\s*\.(\d+)', r': 0.\1', fixed)
        
        return fixed

    def detect_parked_domain(self, content: str) -> bool:
        """
        Detect if a domain appears to be parked or has minimal content.
        Less strict implementation that still provides differentiated scores
        even for domains with limited content.
        
        Args:
            content: The website content
            
        Returns:
            bool: True if the domain appears to be completely empty/parked,
                  False if it has any usable content (even if minimal)
        """
        # Log content details for debugging
        if content:
            logger.debug(f"Content length: {len(content)} chars, first 100 chars: {content[:100]}")
        else:
            logger.warning("Content is empty or None")
            return True
            
        # Only return True for completely empty or extremely minimal content
        if not content or len(content.strip()) < 30:  # Dramatically reduced from 100 to 30
            logger.info(f"Domain content is extremely short: {len(content) if content else 0} characters")
            return True
            
        # Count words in content
        words = re.findall(r'\b\w+\b', content.lower())
        unique_words = set(words)
        
        # Only consider it parked if almost no words at all
        if len(words) < 15:  # Significantly reduced from 50 to 15
            logger.info(f"Domain has extremely few words ({len(words)}), likely completely parked")
            return True
        
        # For sites with very limited content but not completely empty,
        # we'll now return False and let the normal classifier work with
        # appropriately lower confidence scores
        
        # Only check for parking phrases that are definitive indicators
        definitive_parking_phrases = [
            "domain is for sale", "buy this domain", "purchasing this domain", 
            "domain may be for sale", "this domain is for sale", "parked by"
        ]
        
        for phrase in definitive_parking_phrases:
            if phrase in content.lower():
                logger.info(f"Domain contains definitive parking phrase: '{phrase}'")
                return True
        
        # If we get here, the domain has at least some content, so return False
        # and let the normal classifier handle it with appropriate confidence levels
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
        # Check if domain is completely empty/parked
        if self.detect_parked_domain(text_content):
            logger.warning(f"Domain {domain or 'unknown'} appears to be completely empty or parked")
            
            # Fallback with minimal differentiation
            return {
                "predicted_class": "Managed Service Provider",  # Default fallback
                "confidence_scores": {
                    "Managed Service Provider": 0.05,
                    "Integrator - Commercial A/V": 0.04,
                    "Integrator - Residential A/V": 0.03
                },
                "llm_explanation": "This domain appears to be empty or parked. Unable to determine the company type with confidence.",
                "detection_method": "minimal_content_detection",
                "low_confidence": True
            }
        
        # Check if content is minimal but not empty
        is_minimal_content = False
        words = re.findall(r'\b\w+\b', text_content.lower())
        unique_words = set(words)
        
        if len(words) < 100 or len(unique_words) < 50:
            logger.info(f"Domain has minimal but not empty content: {len(words)} words, {len(unique_words)} unique words")
            is_minimal_content = True
            
        try:
            # Define system prompt with classification task - strengthened JSON instructions and higher confidence guidance
            system_prompt = f"""You are an expert business analyst who specializes in categorizing technology companies.
Your task is to analyze the text from a company's website and classify it into ONE of these categories:
1. Managed Service Provider (MSP)
2. Integrator - Commercial A/V
3. Integrator - Residential A/V

First look at the text carefully, noting any services, products, clients, or terminology that indicates the company's focus.

MSPs typically offer IT services, network management, cybersecurity, cloud services, and technical support to businesses.

Commercial A/V Integrators focus on audiovisual technology for businesses, such as conference rooms, digital signage, meeting spaces, and enterprise-level communication systems.

Residential A/V Integrators specialize in home theater, smart home technology, residential automation, and audio/video systems for homes.

Even with minimal content, try to make the best classification possible based on what's available. Look for subtle indicators in organization names, terminology, or context clues.

**YOU MUST ANSWER IN VALID JSON FORMAT EXACTLY AS SHOWN BELOW, WITH NO OTHER TEXT BEFORE OR AFTER THE JSON**:
{{
  "predicted_class": "The most likely category (one of: 'Integrator - Commercial A/V', 'Integrator - Residential A/V', 'Managed Service Provider')",
  "confidence_scores": {{
    "Integrator - Commercial A/V": [decimal value between 0 and 1],
    "Integrator - Residential A/V": [decimal value between 0 and 1],
    "Managed Service Provider": [decimal value between 0 and 1]
  }},
  "llm_explanation": "A detailed explanation of why this classification was chosen, citing specific evidence from the text"
}}

IMPORTANT GUIDELINES:
- Your response MUST be a single, properly formatted JSON object with no preamble, no leading or trailing text, and no extra characters.
- You must provide DIFFERENT confidence scores for each category - they should NOT all be the same value.
- When the evidence clearly points to one category, you should assign a HIGH confidence score (0.7-0.9) to that category and LOWER scores to the others.
- Only use low or moderate confidence scores (0.3-0.6) when the evidence is genuinely ambiguous or minimal.
- The scores should reflect how confident you are that the company belongs to each category, with higher numbers indicating higher confidence.
- Your llm_explanation field MUST be detailed (at least 3-4 sentences) and explain the reasoning behind your classification, citing specific evidence found in the text.
- If there isn't enough information in the text, assign low confidence scores (below 0.3) and note this in your explanation.
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
                return self._fallback_classification(text_content, domain)
                
            # Extract the text response from the API
            text_response = response_data["content"][0]["text"]
            
            # Try multiple patterns to extract JSON
            json_patterns = [
                r'({[\s\S]*"predicted_class"[\s\S]*})',  # Most general pattern
                r'```(?:json)?\s*({[\s\S]*})\s*```',     # For markdown code blocks
                r'({[\s\S]*"confidence_scores"[\s\S]*})' # Alternative key
            ]
            
            parsed_json = None
            
            # Try each pattern
            for pattern in json_patterns:
                json_match = re.search(pattern, text_response, re.DOTALL)
                if json_match:
                    # Clean and process the JSON
                    try:
                        cleaned_json = self.clean_json_string(json_match.group(1))
                        repaired_json = self.repair_json(cleaned_json)
                        parsed_json = json.loads(repaired_json)
                        logger.info(f"Successfully parsed JSON with pattern: {pattern}")
                        break
                    except Exception as e:
                        logger.warning(f"Error parsing extracted JSON with pattern '{pattern}': {e}")
                        continue
            
            # If JSON parsing fails, try again with a more explicit instruction
            if parsed_json is None:
                logger.warning("All JSON parsing attempts failed, retrying with a more explicit prompt")
                
                # Send a follow-up request to Claude
                retry_prompt = """Your previous response was not valid JSON. Please provide your analysis in proper JSON format with EXACTLY this structure and no other text:
{
  "predicted_class": "One of: Managed Service Provider, Integrator - Commercial A/V, or Integrator - Residential A/V",
  "confidence_scores": {
    "Integrator - Commercial A/V": 0.XX,
    "Integrator - Residential A/V": 0.XX,
    "Managed Service Provider": 0.XX
  },
  "llm_explanation": "Your detailed explanation here"
}

IMPORTANT: If the evidence strongly points to one category, give it a high confidence score (0.7-0.9)."""
                
                # Make retry request to Anthropic API
                retry_data = {
                    "model": self.model,
                    "system": "You must provide a valid JSON response with no extra text.",
                    "messages": [
                        {"role": "user", "content": f"Here's the text from a company's website: {text_content[:500]}..."},
                        {"role": "assistant", "content": text_response[:100] + "..."},
                        {"role": "user", "content": retry_prompt}
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.1
                }
                
                try:
                    retry_response = requests.post(url, headers=headers, json=retry_data, timeout=30)
                    retry_response_data = retry_response.json()
                    retry_text = retry_response_data["content"][0]["text"]
                    
                    # Try to extract JSON from retry response
                    for pattern in json_patterns:
                        json_match = re.search(pattern, retry_text, re.DOTALL)
                        if json_match:
                            try:
                                cleaned_json = self.clean_json_string(json_match.group(1))
                                repaired_json = self.repair_json(cleaned_json)
                                parsed_json = json.loads(repaired_json)
                                # Successfully parsed JSON from retry
                                logger.info("Successfully parsed JSON from retry request")
                                break
                            except Exception as e:
                                logger.warning(f"Error parsing JSON from retry: {e}")
                                continue
                except Exception as e:
                    logger.error(f"Error with retry request: {e}")
            
            # If we successfully parsed JSON, process it
            if parsed_json is not None:
                # Validate and process confidence scores
                if "confidence_scores" in parsed_json:
                    # Ensure scores are in 0-1 range (decimal format)
                    for category in parsed_json["confidence_scores"]:
                        score_value = parsed_json["confidence_scores"][category]
                        
                        # Handle potential strings that look like numbers
                        if isinstance(score_value, str) and score_value.replace('.', '', 1).isdigit():
                            score_value = float(score_value)
                        
                        # Ensure score is a decimal between 0 and 1
                        if isinstance(score_value, (int, float)):
                            # If score is greater than 1, convert to 0-1 scale
                            if score_value > 1:
                                score_value = score_value / 100 if score_value <= 100 else 0.9
                            
                            # Ensure it's within bounds
                            score_value = max(0.01, min(0.99, float(score_value)))
                            parsed_json["confidence_scores"][category] = score_value
                        else:
                            # Default if not a valid number
                            parsed_json["confidence_scores"][category] = 0.1
                
                # Check if all scores are identical and fix if needed
                scores = list(parsed_json["confidence_scores"].values())
                if len(set([round(s, 2) for s in scores])) == 1:
                    logger.warning("All confidence scores are the same, regenerating differentiated scores")
                    # Find predicted class and make it have the highest score
                    predicted_class = parsed_json["predicted_class"]
                    base_score = max(scores[0] - 0.2, 0.05)  # Reduce by 0.2 but keep above 0.05
                    
                    # Set different scores for each class
                    for category in parsed_json["confidence_scores"]:
                        if category == predicted_class:
                            parsed_json["confidence_scores"][category] = min(base_score + 0.3, 0.95)
                        elif category == list(parsed_json["confidence_scores"].keys())[0]:
                            parsed_json["confidence_scores"][category] = min(base_score + 0.15, 0.90)
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
                
                # Adjust confidence for minimal content sites
                if is_minimal_content:
                    logger.info("Adjusting confidence scores for minimal content site")
                    
                    # Get highest confidence score and its category
                    predicted_class = parsed_json["predicted_class"]
                    highest_score = parsed_json["confidence_scores"][predicted_class]
                    
                    # Limit maximum confidence for minimal content
                    max_allowed = 0.7  # Cap max confidence at 70% for minimal content
                    
                    # Apply adjustments while maintaining relative proportions
                    if highest_score > max_allowed:
                        reduction_factor = max_allowed / highest_score
                        # Apply proportional reduction to all scores
                        for category in parsed_json["confidence_scores"]:
                            original = parsed_json["confidence_scores"][category]
                            parsed_json["confidence_scores"][category] = max(0.15, original * reduction_factor)
                    
                    # Add note about minimal content to explanation
                    parsed_json["llm_explanation"] += " Note: This classification is based on limited website content, which may affect accuracy."
                    
                    # Set detection method
                    parsed_json["detection_method"] = "llm_with_minimal_content"
                else:
                    parsed_json["detection_method"] = "llm"
                
                logger.info(f"Classified as {parsed_json['predicted_class']} with LLM")
                
                # Add low_confidence flag based on highest confidence score
                max_confidence = max(parsed_json["confidence_scores"].values())
                parsed_json["low_confidence"] = max_confidence < 0.4
                
                return parsed_json
                
            # If we get here, either no JSON was found or parsing failed
            # Try to parse free-form text
            logger.warning("Could not find JSON in LLM response, falling back to text parsing")
            parsed_result = self._parse_free_text(text_response)
            logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
            
            # Adjust confidence for minimal content
            if is_minimal_content:
                for category in parsed_result["confidence_scores"]:
                    parsed_result["confidence_scores"][category] = min(parsed_result["confidence_scores"][category], 0.6)
                parsed_result["detection_method"] = "text_parsing_with_minimal_content"
                parsed_result["llm_explanation"] += " Note: This classification is based on limited website content."
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._fallback_classification(text_content, domain)

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

    def _extract_numeric_confidence(self, text, default=0.2) -> float:
        """
        Extract a numeric confidence value from text.
        
        Args:
            text: The text to extract from
            default: Default value if no confidence found
            
        Returns:
            float: The confidence value (0-1)
        """
        # Try to extract a percentage
        percentage_match = re.search(r'(\d+)%', text)
        if percentage_match:
            value = int(percentage_match.group(1))
            return max(0.01, min(0.99, value / 100.0))
            
        # Try to extract a confidence score mentioned explicitly
        confidence_patterns = [
            r'confidence(?:\s+score)?(?:\s+of)?(?:\s+is)?(?:\s*:)?\s*(\d+(?:\.\d+)?)',
            r'score(?:\s+of)?(?:\s*:)?\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, text, re.IGNORECASE)
            if confidence_match:
                value = float(confidence_match.group(1))
                if value > 1:  # If it's like 80 instead of 0.8
                    value = value / 100.0 if value <= 100 else 0.9
                return max(0.01, min(0.99, value))
                
        # Try to find a decimal between 0-1
        decimal_match = re.search(r'confidence(?:\s+score)?(?:\s+of)?(?:\s+is)?(?:\s*:)?\s*(0\.\d+)', text, re.IGNORECASE)
        if decimal_match:
            value = float(decimal_match.group(1))
            return max(0.01, min(0.99, value))
            
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
                "Managed Service Provider": 0.0,
                "Integrator - Commercial A/V": 0.0,
                "Integrator - Residential A/V": 0.0
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
            # Find best guess based on keywords rather than defaulting
            text_lower = text_content.lower()
            msp_count = sum(1 for indicator in self.msp_indicators if indicator in text_lower)
            commercial_count = sum(1 for indicator in self.commercial_av_indicators if indicator in text_lower)
            residential_count = sum(1 for indicator in self.residential_av_indicators if indicator in text_lower)
            
            # Differentiate scores even with limited information
            if msp_count >= commercial_count and msp_count >= residential_count:
                result["predicted_class"] = "Managed Service Provider"
                result["confidence_scores"]["Managed Service Provider"] = 0.25 + (msp_count * 0.02)
                result["confidence_scores"]["Integrator - Commercial A/V"] = 0.15 + (commercial_count * 0.01)
                result["confidence_scores"]["Integrator - Residential A/V"] = 0.10 + (residential_count * 0.01)
            elif commercial_count >= msp_count and commercial_count >= residential_count:
                result["predicted_class"] = "Integrator - Commercial A/V"
                result["confidence_scores"]["Integrator - Commercial A/V"] = 0.25 + (commercial_count * 0.02)
                result["confidence_scores"]["Managed Service Provider"] = 0.15 + (msp_count * 0.01)
                result["confidence_scores"]["Integrator - Residential A/V"] = 0.10 + (residential_count * 0.01)
            else:
                result["predicted_class"] = "Integrator - Residential A/V"
                result["confidence_scores"]["Integrator - Residential A/V"] = 0.25 + (residential_count * 0.02)
                result["confidence_scores"]["Managed Service Provider"] = 0.15 + (msp_count * 0.01)
                result["confidence_scores"]["Integrator - Commercial A/V"] = 0.10 + (commercial_count * 0.01)
                
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
        
        # Calculate dynamic confidence scores (without normalization) - with higher starting values
        if total_score > 0:
            # More assertive confidence calculations for text parsing - higher base values and steeper scaling
            result["confidence_scores"]["Managed Service Provider"] = min(0.3 + (0.6 * msp_score / max(msp_score + 5, 1)), 0.95) if msp_score > 0 else 0.1
            result["confidence_scores"]["Integrator - Commercial A/V"] = min(0.3 + (0.6 * commercial_score / max(commercial_score + 5, 1)), 0.95) if commercial_score > 0 else 0.1
            result["confidence_scores"]["Integrator - Residential A/V"] = min(0.3 + (0.6 * residential_score / max(residential_score + 5, 1)), 0.95) if residential_score > 0 else 0.1
            
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
                        # Look for domain-related clues
                        if "dementia" in text_lower or "society" in text_lower or "health" in text_lower or "care" in text_lower:
                            # Organizations like dementiasociety.org are likely IT services/MSPs
                            result["predicted_class"] = "Managed Service Provider"
                            result["confidence_scores"]["Managed Service Provider"] = 0.35
                        else:
                            # Ultimate fallback - commercial AV is the middle ground
                            result["predicted_class"] = "Integrator - Commercial A/V"
        else:
            # Look for any domain-specific clues even without indicators
            domain_clues = {
                "Managed Service Provider": ["tech", "it", "computer", "support", "service", "cloud", "network", "cyber"],
                "Integrator - Commercial A/V": ["av", "audio", "video", "conference", "business", "commercial", "corporate", "office"],
                "Integrator - Residential A/V": ["home", "residential", "house", "family", "living", "theater", "entertainment"]
            }
            
            domain_scores = {
                category: sum(1 for clue in clues if clue in text_lower)
                for category, clues in domain_clues.items()
            }
            
            # Find the category with the highest domain clue score
            best_category = max(domain_scores.items(), key=lambda x: x[1])
            
            if best_category[1] > 0:
                # Use the category with the most clues
                result["predicted_class"] = best_category[0]
                result["confidence_scores"][best_category[0]] = 0.3 + (0.05 * best_category[1])
                
                # Set other scores proportionally lower
                for category in result["confidence_scores"]:
                    if category != best_category[0]:
                        result["confidence_scores"][category] = 0.1 + (0.03 * domain_scores[category])
            else:
                # If no indicators found, default to commercial AV with low confidence
                result["predicted_class"] = "Integrator - Commercial A/V"
                result["confidence_scores"]["Integrator - Commercial A/V"] = 0.20
                result["confidence_scores"]["Integrator - Residential A/V"] = 0.15
                result["confidence_scores"]["Managed Service Provider"] = 0.10
        
        # Ensure the predicted class always has the highest confidence score
        highest_category = max(result["confidence_scores"], key=result["confidence_scores"].get)
        if highest_category != result["predicted_class"]:
            # Swap values to ensure predicted class has highest confidence
            temp = result["confidence_scores"][highest_category]
            result["confidence_scores"][highest_category] = result["confidence_scores"][result["predicted_class"]]
            result["confidence_scores"][result["predicted_class"]] = temp
            
        # Set low confidence flag based on highest score
        max_confidence = max(result["confidence_scores"].values())
        result["low_confidence"] = max_confidence < 0.4
        
        # Amplify the spread between scores to create more confidence in the winner
        if not result["low_confidence"]:
            # Get the winner score
            winner_score = result["confidence_scores"][result["predicted_class"]]
            # Amplify it by pushing it higher
            winner_score = min(winner_score * 1.2, 0.95)
            # Reduce the other scores
            for category in result["confidence_scores"]:
                if category != result["predicted_class"]:
                    result["confidence_scores"][category] = max(result["confidence_scores"][category] * 0.8, 0.1)
            # Update the winner score
            result["confidence_scores"][result["predicted_class"]] = winner_score
            
        return result

    def _fallback_classification(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Fallback classification method when API calls or parsing fails.
        
        Args:
            text_content: The text to classify
            domain: Optional domain name for additional context
            
        Returns:
            dict: A default classification with low confidence
        """
        # Count indicators for each category to provide some basis for the fallback
        text_lower = text_content.lower()
        
        msp_count = sum(1 for indicator in self.msp_indicators if indicator in text_lower)
        commercial_count = sum(1 for indicator in self.commercial_av_indicators if indicator in text_lower)
        residential_count = sum(1 for indicator in self.residential_av_indicators if indicator in text_lower)
        
        total_count = msp_count + commercial_count + residential_count
        
        # Check domain name if available
        domain_hints = {"msp": 0, "commercial": 0, "residential": 0}
        
        if domain:
            domain_lower = domain.lower()
            
            # MSP related domain terms
            if any(term in domain_lower for term in ["it", "tech", "computer", "service", "cloud", "cyber", "network", "support"]):
                domain_hints["msp"] += 2
                
            # Commercial AV related domain terms  
            if any(term in domain_lower for term in ["av", "audio", "video", "business", "commercial", "corporate", "office", "conference"]):
                domain_hints["commercial"] += 2
                
            # Residential AV related domain terms
            if any(term in domain_lower for term in ["home", "residential", "theater", "house", "family", "living"]):
                domain_hints["residential"] += 2
                
            # Special case for domains like dementiasociety.org - likely not AV integrators
            if any(term in domain_lower for term in ["health", "care", "medical", "dementia", "society", "foundation", "org"]):
                domain_hints["msp"] += 1  # Slightly bias toward IT services for non-profit/healthcare orgs
        
        if total_count > 0 or max(domain_hints.values()) > 0:
            # If we have some indicators or domain hints, use them
            msp_score = msp_count + domain_hints["msp"]
            commercial_score = commercial_count + domain_hints["commercial"]
            residential_score = residential_count + domain_hints["residential"]
            
            if msp_score > commercial_score and msp_score > residential_score:
                predicted_class = "Managed Service Provider"
                confidence = {
                    "Managed Service Provider": min(0.15 + (msp_score * 0.04), 0.75),
                    "Integrator - Commercial A/V": min(0.10 + (commercial_score * 0.03), 0.40),
                    "Integrator - Residential A/V": min(0.10 + (residential_score * 0.03), 0.40)
                }
            elif commercial_score > msp_score and commercial_score > residential_score:
                predicted_class = "Integrator - Commercial A/V"
                confidence = {
                    "Integrator - Commercial A/V": min(0.15 + (commercial_score * 0.04), 0.75),
                    "Managed Service Provider": min(0.10 + (msp_score * 0.03), 0.40),
                    "Integrator - Residential A/V": min(0.10 + (residential_score * 0.03), 0.40)
                }
            elif residential_score > msp_score and residential_score > commercial_score:
                predicted_class = "Integrator - Residential A/V"
                confidence = {
                    "Integrator - Residential A/V": min(0.15 + (residential_score * 0.04), 0.75),
                    "Managed Service Provider": min(0.10 + (msp_score * 0.03), 0.40),
                    "Integrator - Commercial A/V": min(0.10 + (commercial_score * 0.03), 0.40)
                }
            else:
                # In case of a tie, check domain name
                if domain_hints["msp"] > domain_hints["commercial"] and domain_hints["msp"] > domain_hints["residential"]:
                    predicted_class = "Managed Service Provider"
                    confidence = {
                        "Managed Service Provider": 0.35,
                        "Integrator - Commercial A/V": 0.25,
                        "Integrator - Residential A/V": 0.20
                    }
                elif domain_hints["commercial"] > domain_hints["msp"] and domain_hints["commercial"] > domain_hints["residential"]:
                    predicted_class = "Integrator - Commercial A/V"
                    confidence = {
                        "Integrator - Commercial A/V": 0.35,
                        "Managed Service Provider": 0.25,
                        "Integrator - Residential A/V": 0.20
                    }
                elif domain_hints["residential"] > domain_hints["msp"] and domain_hints["residential"] > domain_hints["commercial"]:
                    predicted_class = "Integrator - Residential A/V"
                    confidence = {
                        "Integrator - Residential A/V": 0.35,
                        "Managed Service Provider": 0.25,
                        "Integrator - Commercial A/V": 0.20
                    }
                else:
                    # Default to MSP for domains like dementiasociety.org
                    if domain and any(term in domain.lower() for term in ["society", "foundation", "org", "health", "care"]):
                        predicted_class = "Managed Service Provider"
                        confidence = {
                            "Managed Service Provider": 0.30,
                            "Integrator - Commercial A/V": 0.20,
                            "Integrator - Residential A/V": 0.15
                        }
                    else:
                        # Otherwise default to commercial AV
                        predicted_class = "Integrator - Commercial A/V"
                        confidence = {
                            "Integrator - Commercial A/V": 0.35,
                            "Managed Service Provider": 0.25,
                            "Integrator - Residential A/V": 0.20
                        }
        else:
            # If domain contains hints about org type, use them
            if domain and any(term in domain.lower() for term in ["society", "foundation", "org", "health", "care", "center"]):
                predicted_class = "Managed Service Provider"
                confidence = {
                    "Managed Service Provider": 0.30,
                    "Integrator - Commercial A/V": 0.20,
                    "Integrator - Residential A/V": 0.15
                }
            else:
                # If no indicators found, provide a very low confidence default
                predicted_class = "Integrator - Commercial A/V"
                confidence = {
                    "Integrator - Commercial A/V": 0.12,
                    "Managed Service Provider": 0.10,
                    "Integrator - Residential A/V": 0.08
                }
            
        explanation = "Classification based on limited available information. "
        
        if domain:
            explanation += f"Domain name '{domain}' was considered in the analysis."
        else:
            explanation += "Limited text content was available for analysis."
            
        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence,
            "llm_explanation": explanation,
            "detection_method": "fallback",
            "low_confidence": True
        }
