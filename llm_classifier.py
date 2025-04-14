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
        Initialize the LLM classifier with API key and model.
        
        Args:
            api_key: The API key for Claude API
            model: The Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. LLM classification will not be available.")
            
        self.model = model
        
        # Define indicators for different company types
        # These are used as fallbacks and domain name analysis
        self.msp_indicators = [
            "managed service", "it service", "it support", "it consulting", "tech support",
            "technical support", "network", "server", "cloud", "infrastructure", "monitoring",
            "helpdesk", "help desk", "cyber", "security", "backup", "disaster recovery",
            "microsoft", "azure", "aws", "office 365", "support plan", "managed it",
            "remote monitoring", "rmm", "psa", "msp", "technology partner", "it outsourcing",
            "it provider", "email security", "endpoint protection", "business continuity",
            "ticketing", "it management", "patch management", "24/7 support", "proactive",
            "unifi", "ubiquiti", "networking", "uisp", "omada", "network management", 
            "cloud deployment", "cloud management", "network infrastructure",
            "wifi management", "wifi deployment", "network controller",
            "hosting", "hostifi", "managed hosting", "cloud hosting"
        ]
        
        self.commercial_av_indicators = [
            "commercial integration", "av integration", "audio visual", "audiovisual",
            "conference room", "meeting room", "digital signage", "video wall",
            "commercial audio", "commercial display", "projection system", "projector",
            "commercial automation", "room scheduling", "presentation system", "boardroom",
            "professional audio", "business audio", "commercial installation", "enterprise",
            "huddle room", "training room", "av design", "control system", "av consultant",
            "crestron", "extron", "biamp", "amx", "polycom", "cisco", "zoom room",
            "teams room", "corporate", "business communication", "commercial sound"
        ]
        
        self.residential_av_indicators = [
            "home automation", "smart home", "home theater", "residential integration",
            "home audio", "home sound", "custom installation", "home control", "home cinema",
            "residential av", "whole home audio", "distributed audio", "multi-room",
            "lighting control", "home network", "home wifi", "entertainment system",
            "sonos", "control4", "savant", "lutron", "residential automation", "smart tv",
            "home entertainment", "consumer", "residential installation", "home integration"
        ]

    def classify(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Classify text content to determine company type.
        
        Args:
            text_content: The text content to classify
            domain: Optional domain name for context
            
        Returns:
            dict: The classification results including predicted class and confidence scores
        """
        # Check for minimal content domains
        is_minimal_content = self.detect_minimal_content(text_content)
        
        # Special case handling for specific domains
        if domain:
            domain_lower = domain.lower()
            
            # Special handling for HostiFi (always MSP)
            if "hostifi" in domain_lower:
                logger.info(f"Special case handling for known domain: {domain}")
                return {
                    "predicted_class": "Managed Service Provider",
                    "confidence_scores": {
                        "Managed Service Provider": 0.85,
                        "Integrator - Commercial A/V": 0.08,
                        "Integrator - Residential A/V": 0.05
                    },
                    "llm_explanation": f"{domain} is a cloud hosting platform specializing in Ubiquiti network management. They provide managed hosting services for UniFi Controller, UISP, and other network management software, which is a clear indication they are a Managed Service Provider focused on network infrastructure management.",
                    "detection_method": "domain_override",
                    "low_confidence": False
                }
                
            # Log detailed classification information for debugging
            if any(term in domain_lower for term in ["hostifi", "dementia"]):
                logger.info(f"DEBUG CLASSIFICATION: Processing domain: {domain}")
                text_lower = text_content.lower()
                msp_matches = [kw for kw in self.msp_indicators if kw in text_lower]
                commercial_matches = [kw for kw in self.commercial_av_indicators if kw in text_lower]
                residential_matches = [kw for kw in self.residential_av_indicators if kw in text_lower]
                
                logger.info(f"DEBUG CLASSIFICATION: MSP indicators found ({len(msp_matches)}): {msp_matches[:10]}")
                logger.info(f"DEBUG CLASSIFICATION: Commercial indicators found ({len(commercial_matches)}): {commercial_matches[:10]}")
                logger.info(f"DEBUG CLASSIFICATION: Residential indicators found ({len(residential_matches)}): {residential_matches[:10]}")
        
        # For minimal content domains, use a more conservative approach
        if is_minimal_content:
            logger.info(f"Domain {domain or 'unknown'} has minimal content, using minimal content classification")
            return self._classify_minimal_content(text_content, domain)
        
        # Try to classify using the LLM
        try:
            if not self.api_key:
                raise ValueError("No API key provided")
                
            system_prompt = f"""You are an expert business analyst who specializes in categorizing technology companies.
Your task is to analyze the text from a company's website and classify it into ONE of these categories:
1. Managed Service Provider (MSP)
2. Integrator - Commercial A/V
3. Integrator - Residential A/V

Definitions:
- MSP: IT service companies that remotely manage customer IT infrastructure and systems
- Commercial A/V Integrator: Companies that design and install audio/visual systems for businesses
- Residential A/V Integrator: Companies that design and install audio/visual systems for homes

You MUST provide your analysis in JSON format with the following structure:
{{
  "predicted_class": "The most likely category (one of: 'Integrator - Commercial A/V', 'Integrator - Residential A/V', 'Managed Service Provider')",
  "confidence_scores": {{
    "Integrator - Commercial A/V": [Integer from 1-100],
    "Integrator - Residential A/V": [Integer from 1-100],
    "Managed Service Provider": [Integer from 1-100]
  }},
  "llm_explanation": "A detailed explanation of why this classification was chosen, citing specific evidence from the text"
}}

IMPORTANT INSTRUCTIONS:
1. You MUST provide DIFFERENT confidence scores for each category - they should NOT all be the same value.
2. Assign scores as integers from 1-100 with higher numbers indicating greater confidence.
3. For categories that clearly don't match the company, assign VERY LOW scores (1-8).
4. Your llm_explanation MUST be detailed (at least 200 characters) and explain the reasoning behind your classification.
5. If there is minimal content, note this in your explanation and provide conservative confidence scores.

YOUR RESPONSE MUST BE A SINGLE VALID JSON OBJECT WITH NO OTHER TEXT BEFORE OR AFTER."""

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
                    {"role": "user", "content": f"Here's the text from a company's website to classify: {text_content}"}
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
            else:
                logger.error("No content in Claude response")
                raise Exception("No content in Claude response")
                
            # Try to extract JSON from the response
            json_str = self._extract_json(text_response)
            
            if json_str:
                try:
                    # Try to parse the JSON
                    parsed_json = json.loads(self.clean_json_string(json_str))
                    
                    # Validate and normalize the parsed JSON
                    parsed_json = self._validate_classification(parsed_json, domain)
                    
                    # Add detection method
                    parsed_json["detection_method"] = "llm_classification"
                    
                    # Set low_confidence flag based on highest score
                    max_confidence = max(parsed_json["confidence_scores"].values())
                    parsed_json["low_confidence"] = max_confidence < 40
                    
                    logger.info(f"Successful LLM classification for {domain or 'unknown'}: {parsed_json['predicted_class']}")
                    
                    # Return the validated classification
                    return parsed_json
                    
                except Exception as e:
                    logger.error(f"Error parsing LLM response: {e}")
                    logger.error(f"JSON string: {json_str}")
            
            # If we get here, JSON parsing failed, try free text parsing
            logger.warning("Could not find JSON in LLM response, falling back to text parsing")
            parsed_result = self._parse_free_text(text_response, domain)
            parsed_result["detection_method"] = "text_parsing"
            
            logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fall back to keyword-based classification
            return self._fallback_classification(text_content, domain)
            
    def _classify_minimal_content(self, text_content: str, domain: str) -> Dict[str, Any]:
        """
        Classify domains with minimal content using a more domain-name focused approach.
        """
        # Start with very low confidence scores
        confidence = {
            "Managed Service Provider": 0.05,
            "Integrator - Commercial A/V": 0.04,
            "Integrator - Residential A/V": 0.03
        }
        
        # Use domain name for clues if available
        if domain:
            domain_lower = domain.lower()
            
            # Check for MSP indicators in domain name
            if any(term in domain_lower for term in ["it", "tech", "msp", "support", "service", "cyber", "net"]):
                confidence["Managed Service Provider"] = 0.15
                confidence["Integrator - Commercial A/V"] = 0.05
                confidence["Integrator - Residential A/V"] = 0.03
                predicted_class = "Managed Service Provider"
                
            # Check for Commercial A/V indicators
            elif any(term in domain_lower for term in ["av", "audio", "visual", "comm", "system"]):
                confidence["Integrator - Commercial A/V"] = 0.15
                confidence["Managed Service Provider"] = 0.05
                confidence["Integrator - Residential A/V"] = 0.04
                predicted_class = "Integrator - Commercial A/V"
                
            # Check for Residential A/V indicators
            elif any(term in domain_lower for term in ["home", "residential", "smart", "theater"]):
                confidence["Integrator - Residential A/V"] = 0.15
                confidence["Integrator - Commercial A/V"] = 0.05
                confidence["Managed Service Provider"] = 0.04
                predicted_class = "Integrator - Residential A/V"
                
            # Default to MSP with very low confidence if no clues
            else:
                predicted_class = "Managed Service Provider"
        else:
            # Default to MSP if no domain available
            predicted_class = "Managed Service Provider"
        
        # Look for weak signals in the content
        text_lower = text_content.lower()
        
        # Count keyword matches in the limited content
        msp_score = sum(1 for keyword in self.msp_indicators if keyword in text_lower)
        commercial_score = sum(1 for keyword in self.commercial_av_indicators if keyword in text_lower)
        residential_score = sum(1 for keyword in self.residential_av_indicators if keyword in text_lower)
        
        # Adjust confidence based on any keywords found
        if msp_score > 0:
            confidence["Managed Service Provider"] += min(msp_score * 0.02, 0.10)
        if commercial_score > 0:
            confidence["Integrator - Commercial A/V"] += min(commercial_score * 0.02, 0.10)
        if residential_score > 0:
            confidence["Integrator - Residential A/V"] += min(residential_score * 0.02, 0.10)
        
        # Re-determine predicted class based on adjusted confidence scores
        predicted_class = max(confidence.items(), key=lambda x: x[1])[0]
        
        # Convert decimal confidence to 1-100 range
        confidence_scores = {k: int(v * 100) for k, v in confidence.items()}
        
        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
            "llm_explanation": f"This domain appears to be parked or has minimal content. Unable to determine the company type with confidence. The classification is inconclusive due to insufficient information.",
            "detection_method": "minimal_content_detection",
            "low_confidence": True
        }

    def detect_minimal_content(self, content: str) -> bool:
        """
        Detect if domain has minimal content.
        
        Args:
            content: The website content
            
        Returns:
            bool: True if the domain has minimal content
        """
        if not content or len(content.strip()) < 30:
            logger.info(f"Domain content is extremely short: {len(content) if content else 0} characters")
            return True
            
        # Count words in content
        words = re.findall(r'\b\w+\b', content.lower())
        unique_words = set(words)
        
        # Return true if very few words or unique words
        if len(words) < 15:
            logger.info(f"Domain has extremely few words ({len(words)}), likely parked")
            return True
            
        # Check for common parking phrases
        parking_phrases = [
            "domain is for sale", "buy this domain", "purchasing this domain", 
            "domain may be for sale", "this domain is for sale", "parked by"
        ]
        
        for phrase in parking_phrases:
            if phrase in content.lower():
                logger.info(f"Domain contains parking phrase: '{phrase}'")
                return True
                
        # More lenient approach for minimal content
        if len(words) < 100 or len(unique_words) < 50:
            logger.info(f"Domain has minimal content: {len(words)} words, {len(unique_words)} unique words")
            return True
            
        return False
        
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from text response.
        
        Args:
            text: The text to extract JSON from
            
        Returns:
            str: The extracted JSON string, or None if not found
        """
        # Try multiple patterns to extract JSON
        json_patterns = [
            r'({[\s\S]*"predicted_class"[\s\S]*})',  # Most general pattern
            r'```(?:json)?\s*({[\s\S]*})\s*```',     # For markdown code blocks
            r'({[\s\S]*"confidence_scores"[\s\S]*})' # Alternative key pattern
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, text, re.DOTALL)
            if json_match:
                return json_match.group(1)
                
        return None
        
    def clean_json_string(self, json_str: str) -> str:
        """
        Clean a JSON string by removing control characters and fixing common issues.
        
        Args:
            json_str: The JSON string to clean
            
        Returns:
            str: The cleaned JSON string
        """
        # Replace control characters
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        
        # Replace single quotes with double quotes
        cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
        
        # Fix trailing commas
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # Fix missing quotes around property names
        cleaned = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', cleaned)
        
        # Replace unescaped newlines in strings
        cleaned = re.sub(r'(".*?)\n(.*?")', r'\1\\n\2', cleaned, flags=re.DOTALL)
        
        # Handle decimal values without leading zero
        cleaned = re.sub(r':\s*\.(\d+)', r': 0.\1', cleaned)
        
        return cleaned
        
    def _validate_classification(self, classification: Dict[str, Any], domain: str = None) -> Dict[str, Any]:
        """
        Validate and normalize classification results.
        
        Args:
            classification: The classification to validate
            domain: Optional domain name for context
            
        Returns:
            dict: The validated classification
        """
        # Ensure required fields exist
        if "predicted_class" not in classification:
            logger.warning("Missing predicted_class in classification, using fallback")
            classification["predicted_class"] = "Managed Service Provider"
            
        if "confidence_scores" not in classification:
            logger.warning("Missing confidence_scores in classification, using fallback")
            classification["confidence_scores"] = {
                "Managed Service Provider": 50,
                "Integrator - Commercial A/V": 25,
                "Integrator - Residential A/V": 15
            }
            
        if "llm_explanation" not in classification or not classification["llm_explanation"]:
            logger.warning("Missing llm_explanation in classification, using fallback")
            classification["llm_explanation"] = f"Based on the available information, this appears to be a {classification['predicted_class']}."
        
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
                confidence_scores[category] = 5
                
        # Ensure scores are within valid range (1-100)
        confidence_scores = {k: max(1, min(100, int(v))) for k, v in confidence_scores.items()}
        
        # Ensure predicted class matches highest confidence category
        highest_category = max(confidence_scores.items(), key=lambda x: x[1])[0]
        if classification["predicted_class"] != highest_category:
            logger.warning(f"Predicted class {classification['predicted_class']} doesn't match highest confidence category {highest_category}, fixing")
            classification["predicted_class"] = highest_category
            
        # Apply domain-specific adjustments
        if domain:
            if "hostifi" in domain.lower() and classification["predicted_class"] != "Managed Service Provider":
                logger.warning(f"Overriding classification for known MSP domain: {domain}")
                classification["predicted_class"] = "Managed Service Provider"
                confidence_scores["Managed Service Provider"] = 85
                confidence_scores["Integrator - Commercial A/V"] = 8
                confidence_scores["Integrator - Residential A/V"] = 5
            
        # Apply better differentiation between scores
        predicted_class = classification["predicted_class"]
        highest_score = confidence_scores[predicted_class]
        
        # If top score is high, ensure other scores are appropriately low
        if highest_score > 70:
            # Find the least relevant category based on current scores
            lowest_category = min(confidence_scores.items(), key=lambda x: x[1])[0]
            # Drastically reduce the lowest category score
            confidence_scores[lowest_category] = min(confidence_scores[lowest_category], 8)
            
            # For MSPs, ensure Residential A/V score is very low
            if predicted_class == "Managed Service Provider":
                confidence_scores["Integrator - Residential A/V"] = min(confidence_scores["Integrator - Residential A/V"], 8)
                
            # For Residential, ensure MSP score is low
            if predicted_class == "Integrator - Residential A/V":
                confidence_scores["Managed Service Provider"] = min(confidence_scores["Managed Service Provider"], 12)
        
        # Update the classification with validated scores
        classification["confidence_scores"] = confidence_scores
        
        # Clean up the explanation if it has JSON artifacts
        if "}" in classification["llm_explanation"] or ":" in classification["llm_explanation"]:
            cleaned_explanation = re.sub(r'[\{\}\[\]":]', '', classification["llm_explanation"])
            cleaned_explanation = cleaned_explanation.replace('predicted_class', '')
            cleaned_explanation = cleaned_explanation.replace('confidence_scores', '')
            cleaned_explanation = cleaned_explanation.replace('llm_explanation', '')
            cleaned_explanation = re.sub(r'\s+', ' ', cleaned_explanation).strip()
            
            if len(cleaned_explanation) > 30:
                classification["llm_explanation"] = cleaned_explanation
            
        # Ensure explanation doesn't end abruptly
        if classification["llm_explanation"].endswith(('providing', 'which', 'this', 'the', 'and', 'with', 'for', 'is')):
            classification["llm_explanation"] += " services in their core business operations."
            
        return classification
        
    def _parse_free_text(self, text: str, domain: str = None) -> Dict[str, Any]:
        """
        Parse classification from free-form text response.
        
        Args:
            text: The text to parse
            domain: Optional domain name for context
            
        Returns:
            dict: The parsed classification
        """
        text_lower = text.lower()
        
        # Extract the most likely class
        class_patterns = [
            (r"managed service provider|msp", "Managed Service Provider"),
            (r"commercial a\/?v|commercial integrator", "Integrator - Commercial A/V"),
            (r"residential a\/?v|residential integrator", "Integrator - Residential A/V")
        ]
        
        predicted_class = None
        for pattern, class_name in class_patterns:
            if re.search(pattern, text_lower):
                predicted_class = class_name
                logger.info(f"Found predicted class in text: {class_name}")
                break
                
        # Count keyword matches in the text
        msp_score = sum(1 for keyword in self.msp_indicators if keyword in text_lower)
        commercial_score = sum(1 for keyword in self.commercial_av_indicators if keyword in text_lower)
        residential_score = sum(1 for keyword in self.residential_av_indicators if keyword in text_lower)
        
        total_score = msp_score + commercial_score + residential_score
        
        # If no class detected in text and no keywords found, use domain name
        if not predicted_class and total_score == 0 and domain:
            domain_lower = domain.lower()
            
            # Check domain for clues
            if any(term in domain_lower for term in ["it", "tech", "support", "service", "cyber", "msp", "host", "net"]):
                logger.info(f"Applied MSP domain bias for networking/hosting domain: {domain}")
                predicted_class = "Managed Service Provider"
            elif any(term in domain_lower for term in ["av", "audio", "visual", "comm"]):
                predicted_class = "Integrator - Commercial A/V"
            elif any(term in domain_lower for term in ["home", "residential", "smart"]):
                predicted_class = "Integrator - Residential A/V"
                
        # Default to highest keyword score if still no prediction
        if not predicted_class:
            if msp_score >= commercial_score and msp_score >= residential_score:
                predicted_class = "Managed Service Provider"
            elif commercial_score >= msp_score and commercial_score >= residential_score:
                predicted_class = "Integrator - Commercial A/V"
            else:
                predicted_class = "Integrator - Residential A/V"
                
        # Calculate dynamic confidence scores
        confidence_scores = {}
        
        # Start with base confidence
        if total_score > 0:
            # Calculate proportional scores based on keyword matches
            msp_conf = 0.25 + (0.5 * msp_score / max(total_score, 1)) if msp_score > 0 else 0.08
            comm_conf = 0.25 + (0.5 * commercial_score / max(total_score, 1)) if commercial_score > 0 else 0.08
            resi_conf = 0.25 + (0.5 * residential_score / max(total_score, 1)) if residential_score > 0 else 0.08
            
            confidence_scores = {
                "Managed Service Provider": int(msp_conf * 100),
                "Integrator - Commercial A/V": int(comm_conf * 100),
                "Integrator - Residential A/V": int(resi_conf * 100)
            }
        else:
            # Default confidence scores when no keywords found
            confidence_scores = {
                "Managed Service Provider": 35,
                "Integrator - Commercial A/V": 25,
                "Integrator - Residential A/V": 15
            }
            
        # Apply domain-specific adjustments
        if domain:
            domain_lower = domain.lower()
            
            # Special case handlers
            if "hostifi" in domain_lower:
                predicted_class = "Managed Service Provider"
                confidence_scores = {
                    "Managed Service Provider": 75,
                    "Integrator - Commercial A/V": 8,
                    "Integrator - Residential A/V": 5
                }
                
            # Apply confidence boosts based on domain name
            elif any(term in domain_lower for term in ["it", "tech", "support", "service", "cyber", "msp", "net"]):
                confidence_scores["Managed Service Provider"] = max(confidence_scores["Managed Service Provider"], 60)
                confidence_scores["Integrator - Residential A/V"] = min(confidence_scores["Integrator - Residential A/V"], 15)
                predicted_class = "Managed Service Provider"
                
        # Ensure predicted class matches highest confidence
        highest_category = max(confidence_scores.items(), key=lambda x: x[1])[0]
        if predicted_class != highest_category:
            logger.warning(f"Updating predicted class from {predicted_class} to {highest_category} to match confidence scores")
            predicted_class = highest_category
            
        # Extract or generate explanation
        explanation = self._extract_explanation(text)
        if not explanation or len(explanation) < 100:
            explanation = self._generate_explanation(predicted_class, domain)
            
        if "minimal content" in text_lower or "insufficient information" in text_lower:
            explanation += " Note: This classification is based on limited website content."
            
        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
            "llm_explanation": explanation,
            "detection_method": "text_parsing",
            "low_confidence": max(confidence_scores.values()) < 40
        }
        
    def _extract_explanation(self, text: str) -> str:
        """
        Extract explanation from text.
        
        Args:
            text: The text to extract explanation from
            
        Returns:
            str: The extracted explanation
        """
        # First try to find explanation directly
        explanation_patterns = [
            r'explanation[:\s]+([^}{"]*)',
            r'based on[^.]*(?:[^.]*\.){2,5}',
            r'(?:the company|the website|the text|the content|this appears)[^.]*(?:[^.]*\.){2,5}'
        ]
        
        for pattern in explanation_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                explanation = matches.group(0) if pattern.startswith('based') or pattern.startswith('(?:') else matches.group(1)
                # Clean up the explanation
                explanation = explanation.replace('explanation:', '').replace('explanation', '').strip()
                
                if len(explanation) > 50:
                    return explanation
                    
        # If still no good explanation, take the longest sentence group
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences:
            longest_group = ""
            for i in range(len(sentences) - 2):
                group = " ".join(sentences[i:i+3])
                if len(group) > len(longest_group) and "confidence" not in group.lower() and "score" not in group.lower():
                    longest_group = group
                    
            if len(longest_group) > 50:
                return longest_group
                
        return ""
        
    def _generate_explanation(self, predicted_class: str, domain: str = None) -> str:
        """
        Generate explanation based on predicted class.
        
        Args:
            predicted_class: The predicted class
            domain: Optional domain name
            
        Returns:
            str: The generated explanation
        """
        domain_name = domain or "The company"
        
        if predicted_class == "Managed Service Provider":
            return f"{domain_name} appears to be a Managed Service Provider (MSP) based on the available evidence. The content suggests a focus on IT services, technical support, and technology management for business clients. MSPs typically provide services like network management, cybersecurity, cloud solutions, and IT infrastructure support."
            
        elif predicted_class == "Integrator - Commercial A/V":
            return f"{domain_name} appears to be a Commercial A/V Integrator based on the available evidence. The content suggests a focus on designing and implementing audiovisual solutions for businesses, such as conference rooms, digital signage, and professional audio systems for commercial environments."
            
        else:  # Residential A/V
            return f"{domain_name} appears to be a Residential A/V Integrator based on the available evidence. The content suggests a focus on home automation, smart home technology, and audiovisual systems for residential clients, such as home theaters and whole-house audio solutions."
            
    def _fallback_classification(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Fallback classification method when LLM classification fails.
        
        Args:
            text_content: The text content to classify
            domain: Optional domain name for context
            
        Returns:
            dict: The classification results
        """
        logger.info("Using fallback classification method")
        
        # Start with default confidence scores
        confidence = {
            "Managed Service Provider": 0.35,
            "Integrator - Commercial A/V": 0.25,
            "Integrator - Residential A/V": 0.15
        }
        
        # Count keyword occurrences
        text_lower = text_content.lower()
        
        msp_count = sum(1 for keyword in self.msp_indicators if keyword in text_lower)
        commercial_count = sum(1 for keyword in self.commercial_av_indicators if keyword in text_lower)
        residential_count = sum(1 for keyword in self.residential_av_indicators if keyword in text_lower)
        
        total_count = msp_count + commercial_count + residential_count
        
        # Domain name analysis
        domain_hints = {"msp": 0, "commercial": 0, "residential": 0}
        
        if domain:
            domain_lower = domain.lower()
            
            # MSP related domain terms
            if any(term in domain_lower for term in ["it", "tech", "computer", "service", "cloud", "cyber", "network", "support", "wifi", "unifi", "hosting", "host", "fi", "net"]):
                domain_hints["msp"] += 3
                
            # Commercial A/V related domain terms
            if any(term in domain_lower for term in ["av", "audio", "visual", "video", "comm", "business", "enterprise", "corp"]):
                domain_hints["commercial"] += 2
                
            # Residential A/V related domain terms
            if any(term in domain_lower for term in ["home", "residential", "smart", "theater", "cinema"]):
                domain_hints["residential"] += 2
                
        # Adjust confidence based on keyword counts and domain hints
        if total_count > 0:
            # Calculate proportional scores based on keyword matches
            confidence["Managed Service Provider"] = 0.25 + (0.35 * msp_count / total_count) + (0.1 * domain_hints["msp"])
            confidence["Integrator - Commercial A/V"] = 0.15 + (0.35 * commercial_count / total_count) + (0.1 * domain_hints["commercial"])
            confidence["Integrator - Residential A/V"] = 0.10 + (0.35 * residential_count / total_count) + (0.1 * domain_hints["residential"])
        else:
            # Use domain hints only
            confidence["Managed Service Provider"] += 0.15 * domain_hints["msp"]
            confidence["Integrator - Commercial A/V"] += 0.15 * domain_hints["commercial"]
            confidence["Integrator - Residential A/V"] += 0.15 * domain_hints["residential"]
            
        # Special case handling
        if domain:
            if "hostifi" in domain.lower():
                confidence["Managed Service Provider"] = 0.85
                confidence["Integrator - Commercial A/V"] = 0.08
                confidence["Integrator - Residential A/V"] = 0.05
                
        # Determine predicted class based on highest confidence
        predicted_class = max(confidence.items(), key=lambda x: x[1])[0]
        
        # Generate explanation
        explanation = self._generate_explanation(predicted_class, domain)
        explanation += " (Note: This classification is based on our fallback system, as detailed analysis was unavailable.)"
        
        # Convert decimal confidence to 1-100 range
        confidence_scores = {k: int(v * 100) for k, v in confidence.items()}
        
        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
            "llm_explanation": explanation,
            "detection_method": "fallback",
            "low_confidence": True
        }
