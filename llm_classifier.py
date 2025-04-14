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
        
        # Domain parking and minimal content phrases
        self.parking_phrases = [
            "domain is for sale", "buy this domain", "purchase this domain", 
            "domain may be for sale", "check back soon", "coming soon",
            "website coming soon", "site is under construction", "under construction",
            "parked domain", "web hosting", "domain registration", "godaddy",
            "namecheap", "bluehost", "hostgator", "domain parking", "this webpage is parked",
            "this domain is parked", "domain parking page"
        ]
    
    def detect_parked_domain(self, text: str) -> bool:
        """Detect if a domain appears to be parked or has minimal content."""
        if not text or len(text.strip()) < 100:
            logger.info("Domain has very little content (less than 100 chars)")
            return True
            
        lower_text = text.lower()
        
        # Check for parking phrases
        for phrase in self.parking_phrases:
            if phrase in lower_text:
                logger.info(f"Domain appears to be parked (detected '{phrase}')")
                return True
        
        # Check if content is too generic (not enough specific terms)
        word_count = len(re.findall(r'\b\w+\b', text))
        unique_words = len(set(re.findall(r'\b\w+\b', lower_text)))
        
        if word_count < 50:
            logger.info(f"Domain has only {word_count} words, likely minimal content")
            return True
            
        if unique_words < 30:
            logger.info(f"Domain has only {unique_words} unique words, likely generic content")
            return True
        
        # Check for lack of company-specific content
        business_terms = ['company', 'business', 'service', 'product', 'client', 'customer', 'contact', 'about us']
        if not any(term in lower_text for term in business_terms):
            logger.info("Domain lacks basic business terms, likely not a real company site")
            return True
        
        return False

    def clean_json_string(self, json_str):
        """Clean a JSON string by removing control characters and fixing common issues."""
        # Replace common control characters
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        # Fix unescaped quotes within strings (but not in valid syntax locations)
        cleaned = re.sub(r'([^\\])"([^{}\[\],:"\\])', r'\1\"\2', cleaned)
        return cleaned
    
    def classify(self, text: str, domain: str = None) -> Dict[str, Any]:
        """Classify website text using Claude.
        
        Args:
            text: The website content to classify
            domain: Optional domain name for context
            
        Returns:
            Dict containing classification results
        """
        # First check if this is a parked domain or has minimal content
        if self.detect_parked_domain(text):
            logger.warning(f"Domain {domain} appears to be parked or has minimal content")
            return {
                "predicted_class": "Unknown",
                "confidence_scores": {
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5,
                    "Managed Service Provider": 5
                },
                "max_confidence": 5,
                "llm_explanation": "This domain appears to be parked or has minimal content. It may be a placeholder, under construction, or for sale. There is insufficient information to determine the company type.",
                "detection_method": "minimal_content_detection"
            }
            
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

Analyze the provided website content carefully. You must:
1. Use the specific characteristics described above to determine the company type
2. Assign confidence scores (from 1-100) for EACH category based on the evidence found
3. Provide DIFFERENT confidence scores for each category - they should not all be the same
4. Base your scores on the STRENGTH of evidence, not preconceived biases
5. Write a detailed explanation citing SPECIFIC evidence from the text

Your response must be a JSON object with the following structure:
{
  "predicted_class": "The category with the highest confidence score",
  "confidence_scores": {
    "Integrator - Commercial A/V": Integer from 1-100,
    "Integrator - Residential A/V": Integer from 1-100,
    "Managed Service Provider": Integer from 1-100
  },
  "llm_explanation": "Your detailed explanation with specific evidence from the text"
}

If the website has insufficient content, assign LOW scores to all categories (below 30) and explain the lack of clear evidence.
"""

            user_prompt = f"Based on the following website content{domain_context}, classify the company and explain your reasoning. Remember to assign confidence scores from 1-100 for each category based on specific evidence found:\n\n{text[:15000]}"

            request_data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1500,
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
                            try:
                                # Try to find JSON-like structure in the response
                                json_match = re.search(r'({.*?"predicted_class".*?})', text_content, re.DOTALL)
                                
                                if not json_match:
                                    logger.warning("Could not find JSON in LLM response, falling back to text parsing")
                                    # Try to parse free-form text
                                    parsed_result = self._parse_free_text(text_content)
                                    logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
                                    return parsed_result
                                
                                # Try to clean and parse the JSON
                                try:
                                    json_str = json_match.group(1)
                                    cleaned_json = self.clean_json_string(json_str)
                                    parsed_json = json.loads(cleaned_json)
                                except Exception as e:
                                    logger.error(f"Error parsing cleaned JSON: {e}")
                                    # Fall back to text parsing
                                    parsed_result = self._parse_free_text(text_content)
                                    return parsed_result
                                
                                # Validate and process JSON fields
                                # Ensure all required fields are present
                                if "predicted_class" not in parsed_json:
                                    logger.warning("Missing predicted_class in JSON response")
                                    parsed_json["predicted_class"] = "Unknown"
                                
                                # Handle confidence scores
                                if "confidence_scores" not in parsed_json:
                                    logger.warning("Missing confidence_scores in JSON response")
                                    # Generate confidence scores from text
                                    parsed_json["confidence_scores"] = self._extract_confidence_scores(text_content)
                                else:
                                    # Check and normalize confidence scores to ensure they're integers 1-100
                                    confidence_scores = parsed_json["confidence_scores"]
                                    normalized_scores = {}
                                    
                                    for category, score in confidence_scores.items():
                                        # Convert float to int if needed
                                        if isinstance(score, float):
                                            # If score appears to be 0-1 scale, convert to 1-100
                                            if 0 <= score <= 1:
                                                score = int(score * 100)
                                            else:
                                                score = int(score)
                                        
                                        # Ensure score is within 1-100 range
                                        score = max(1, min(100, score))
                                        normalized_scores[category] = score
                                    
                                    parsed_json["confidence_scores"] = normalized_scores
                                
                                # Check if all confidence scores are the same
                                scores = list(parsed_json["confidence_scores"].values())
                                if len(scores) > 1 and all(score == scores[0] for score in scores):
                                    logger.warning("All confidence scores are the same, adjusting for differentiation")
                                    # Slightly adjust scores to ensure they're different
                                    categories = list(parsed_json["confidence_scores"].keys())
                                    parsed_json["confidence_scores"][categories[0]] = max(1, min(100, scores[0] - 1))
                                    parsed_json["confidence_scores"][categories[2]] = max(1, min(100, scores[0] + 1))
                                
                                # Ensure llm_explanation is included
                                if "llm_explanation" not in parsed_json or not parsed_json["llm_explanation"]:
                                    logger.warning("Missing llm_explanation in JSON response")
                                    # Extract reasoning from the full text
                                    parsed_json["llm_explanation"] = self._extract_explanation(text_content)
                                
                                # Calculate max confidence score and ensure predicted_class is consistent
                                confidence_scores = parsed_json["confidence_scores"]
                                if confidence_scores:
                                    max_class = max(confidence_scores.items(), key=lambda x: x[1])
                                    parsed_json["predicted_class"] = max_class[0]
                                    parsed_json["max_confidence"] = max_class[1]
                                else:
                                    logger.warning("No confidence scores available")
                                    parsed_json["max_confidence"] = 50  # Default confidence
                                
                                # Set detection method
                                parsed_json["detection_method"] = "llm_classification"
                                
                                # Log the explanation length for debugging
                                if "llm_explanation" in parsed_json:
                                    logger.info(f"Explanation length: {len(parsed_json['llm_explanation'])}")
                                
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
                    "Integrator - Commercial A/V": 5,
                    "Integrator - Residential A/V": 5,
                    "Managed Service Provider": 5
                },
                "max_confidence": 5,
                "llm_explanation": f"Classification failed due to an error: {str(e)}",
                "detection_method": "llm_classification_failed"
            }
    
    def _parse_free_text(self, text: str) -> Dict[str, Any]:
        """Parse classification from free-form text response."""
        result = {
            "confidence_scores": {
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Managed Service Provider": 0
            },
            "llm_explanation": ""
        }
        
        # Check for indications of minimal content
        minimal_content_phrases = [
            "not enough information", "insufficient information", "insufficient content",
            "minimal content", "limited content", "not enough content", "lack of content",
            "generic content", "generic landing page", "placeholder", "parked domain"
        ]
        
        if any(phrase in text.lower() for phrase in minimal_content_phrases):
            logger.info("LLM response indicates minimal content")
            result["predicted_class"] = "Unknown"
            result["confidence_scores"]["Integrator - Commercial A/V"] = 5
            result["confidence_scores"]["Integrator - Residential A/V"] = 5
            result["confidence_scores"]["Managed Service Provider"] = 5
            result["max_confidence"] = 5
            result["llm_explanation"] = "There is insufficient content on this website to determine the company type. The website may be a generic landing page, parked domain, or under construction."
            result["detection_method"] = "minimal_content_detection"
            return result
        
        # First, examine the explanation to identify the main classification
        lower_text = text.lower()
        
        # Look for confidence score statements
        # Pattern like "70% confidence" or "confidence score of 70" or "70/100"
        commercial_confidence = self._extract_numeric_confidence("commercial a/v", lower_text)
        residential_confidence = self._extract_numeric_confidence("residential a/v", lower_text)
        msp_confidence = self._extract_numeric_confidence("managed service provider", lower_text)
        
        # Check if we found explicit confidence scores
        scores_found = (commercial_confidence > 0 or residential_confidence > 0 or msp_confidence > 0)
        
        if not scores_found:
            # Look for clear statements about the classification
            msp_indicators = ["managed service provider", "msp", "it service", "it support", 
                         "classify as a managed service", "classify it as a managed service",
                         "classify as an msp", "classify it as an msp", "would classify it as a managed service",
                         "would classify it as an msp", "is a managed service provider", "is an msp",
                         "appears to be a managed service provider", "appears to be an msp"]
                         
            commercial_indicators = ["commercial a/v", "commercial av", "commercial integrator",
                               "classify as a commercial", "classify it as a commercial",
                               "is a commercial a/v integrator", "is a commercial av integrator",
                               "appears to be a commercial a/v", "appears to be a commercial av"]
                               
            residential_indicators = ["residential a/v", "residential av", "home theater", 
                                "classify as a residential", "classify it as a residential",
                                "is a residential a/v integrator", "is a residential av integrator",
                                "appears to be a residential a/v", "appears to be a residential av"]
            
            # Count indicator matches for each category
            msp_score = sum(1 for indicator in msp_indicators if indicator in lower_text)
            commercial_score = sum(1 for indicator in commercial_indicators if indicator in lower_text)
            residential_score = sum(1 for indicator in residential_indicators if indicator in lower_text)
            
            # Calculate total score for reference
            total_score = msp_score + commercial_score + residential_score
            
            # Convert counts to percentage-based scores (1-100)
            if total_score > 0:
                # Calculate base scores plus proportion of matches
                base_score = 20  # Minimum score when an indicator is matched
                remaining_points = 60  # Points to distribute based on proportion
                
                commercial_confidence = base_score if commercial_score > 0 else 10
                residential_confidence = base_score if residential_score > 0 else 10
                msp_confidence = base_score if msp_score > 0 else 10
                
                if commercial_score > 0:
                    commercial_confidence += int(remaining_points * (commercial_score / total_score))
                if residential_score > 0:
                    residential_confidence += int(remaining_points * (residential_score / total_score))
                if msp_score > 0:
                    msp_confidence += int(remaining_points * (msp_score / total_score))
            else:
                # No indicators found, assign default scores
                # Look for keywords related to each category
                keywords = {
                    "Managed Service Provider": ["it", "server", "network", "support", "security", "cloud", "managed", "monitoring"],
                    "Integrator - Commercial A/V": ["business", "conference", "office", "corporation", "enterprise", "presentation"],
                    "Integrator - Residential A/V": ["home", "residential", "theater", "family", "living", "smart home"]
                }
                
                # Count keywords for each category
                keyword_counts = {
                    "Managed Service Provider": sum(1 for word in keywords["Managed Service Provider"] if word in lower_text),
                    "Integrator - Commercial A/V": sum(1 for word in keywords["Integrator - Commercial A/V"] if word in lower_text),
                    "Integrator - Residential A/V": sum(1 for word in keywords["Integrator - Residential A/V"] if word in lower_text)
                }
                
                total_keywords = sum(keyword_counts.values())
                if total_keywords > 0:
                    # Assign scores based on keyword frequency
                    base_score = 15
                    remaining_points = 45
                    
                    msp_confidence = base_score + int(remaining_points * (keyword_counts["Managed Service Provider"] / total_keywords))
                    commercial_confidence = base_score + int(remaining_points * (keyword_counts["Integrator - Commercial A/V"] / total_keywords))
                    residential_confidence = base_score + int(remaining_points * (keyword_counts["Integrator - Residential A/V"] / total_keywords))
                else:
                    # Truly no indicators, assign different but low scores
                    msp_confidence = 35
                    commercial_confidence = 33
                    residential_confidence = 32
        
        # Set the final scores
        result["confidence_scores"]["Managed Service Provider"] = msp_confidence
        result["confidence_scores"]["Integrator - Commercial A/V"] = commercial_confidence
        result["confidence_scores"]["Integrator - Residential A/V"] = residential_confidence
        
        # Set the predicted class
        max_class = max(result["confidence_scores"].items(), key=lambda x: x[1])
        result["predicted_class"] = max_class[0]
        result["max_confidence"] = max_class[1]
        
        # Extract explanation - try to get a substantial one
        result["llm_explanation"] = self._extract_explanation(text)
        result["detection_method"] = "text_classification"
        
        return result
    
    def _extract_numeric_confidence(self, category_term, text):
        """Extract numeric confidence scores for a specific category from text."""
        # Try to find patterns like:
        # - "70% confidence that it's a [category]"
        # - "[category] with 70% confidence"
        # - "confidence score of 70 for [category]"
        # - "[category]: 70/100"
        
        patterns = [
            # 70% confidence
            rf"{category_term}.*?(\d+)%",
            rf"(\d+)%.*?{category_term}",
            # confidence of 70
            rf"{category_term}.*?confidence.*?(\d+)",
            rf"confidence.*?(\d+).*?{category_term}",
            # score of 70
            rf"{category_term}.*?score.*?(\d+)",
            rf"score.*?(\d+).*?{category_term}",
            # 70/100 format
            rf"{category_term}.*?(\d+)/100",
            rf"(\d+)/100.*?{category_term}",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = int(match.group(1))
                    # Ensure score is within 1-100 range
                    return max(1, min(100, score))
                except (ValueError, IndexError):
                    pass
        
        return 0  # No confidence score found
    
    def _extract_confidence_scores(self, text: str) -> Dict[str, int]:
        """Extract confidence scores from text response."""
        scores = {
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0,
            "Managed Service Provider": 0
        }
        
        # Check for insufficient content indicators
        insufficient_content_indicators = [
            "not enough information", "insufficient information", "insufficient content",
            "minimal content", "limited content", "not enough content", "lack of content",
            "generic content", "generic landing page", "placeholder", "parked domain"
        ]
        
        if any(indicator in text.lower() for indicator in insufficient_content_indicators):
            # Set very low confidence for all categories
            scores = {
                "Integrator - Commercial A/V": 5,
                "Integrator - Residential A/V": 5,
                "Managed Service Provider": 5
            }
            return scores
        
        # Try to extract explicit confidence scores
        lower_text = text.lower()
        scores["Integrator - Commercial A/V"] = self._extract_numeric_confidence("commercial a/v", lower_text)
        scores["Integrator - Residential A/V"] = self._extract_numeric_confidence("residential a/v", lower_text)
        scores["Managed Service Provider"] = self._extract_numeric_confidence("managed service provider", lower_text)
        
        # Check if no scores were extracted or if all scores are the same
        all_scores_same = len(set(scores.values())) <= 1
        no_scores = all(score == 0 for score in scores.values())
        
        if no_scores or all_scores_same:
            # Analyze the text content to determine confidence scores
            # Count keyword mentions to differentiate scores
            keywords = {
                "Managed Service Provider": ["it", "server", "network", "support", "security", "cloud", "managed", "monitoring"],
                "Integrator - Commercial A/V": ["business", "conference", "office", "corporation", "enterprise", "presentation"],
                "Integrator - Residential A/V": ["home", "residential", "theater", "family", "living", "smart home"]
            }
            
            # Count keywords for each category
            keyword_counts = {
                "Managed Service Provider": sum(1 for word in keywords["Managed Service Provider"] if word in lower_text),
                "Integrator - Commercial A/V": sum(1 for word in keywords["Integrator - Commercial A/V"] if word in lower_text),
                "Integrator - Residential A/V": sum(1 for word in keywords["Integrator - Residential A/V"] if word in lower_text)
            }
            
            total_keywords = sum(keyword_counts.values())
            if total_keywords > 0:
                # Assign scores based on keyword frequency
                base_score = 15
                remaining_points = 45
                
                scores["Managed Service Provider"] = base_score + int(remaining_points * (keyword_counts["Managed Service Provider"] / total_keywords))
                scores["Integrator - Commercial A/V"] = base_score + int(remaining_points * (keyword_counts["Integrator - Commercial A/V"] / total_keywords))
                scores["Integrator - Residential A/V"] = base_score + int(remaining_points * (keyword_counts["Integrator - Residential A/V"] / total_keywords))
            else:
                # No keywords found, assign different but low scores
                scores["Managed Service Provider"] = 35
                scores["Integrator - Commercial A/V"] = 33
                scores["Integrator - Residential A/V"] = 32
        
        # Final check to ensure scores are different
        if len(set(scores.values())) <= 1:
            # Force differentiation with small adjustments
            scores["Managed Service Provider"] += 3
            scores["Integrator - Commercial A/V"] += 1
        
        return scores
    
    def _extract_explanation(self, text: str) -> str:
        """Extract a detailed explanation from the text response."""
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
                    # Get the text after the marker
                    explanation_text = parts[1].strip()
                    # Try to find where the explanation ends (before another marker or JSON)
                    end_markers = ['{', '}', '"predicted_class"']
                    for end_marker in end_markers:
                        if end_marker in explanation_text:
                            explanation_text = explanation_text.split(end_marker)[0]
                    
                    explanation = explanation_text
                    break
        
        # If no marker found, extract meaningful sentences
        if not explanation or len(explanation) < 50:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            meaningful_sentences = []
            
            # Keywords that might indicate reasoning or explanation
            reasoning_keywords = [
                "because", "since", "reason", "evidence", "indicates", "suggests",
                "mentions", "references", "focuses on", "specializes in", "offers",
                "provides", "features", "highlights", "exhibits", "demonstrates",
                "I found", "I noticed", "I observed", "appears to be", "based on",
                "website contains", "content shows", "text mentions"
            ]
            
            # Collect sentences that might be part of an explanation
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and any(keyword in sentence.lower() for keyword in reasoning_keywords):
                    meaningful_sentences.append(sentence)
            
            # If we found some reasoning sentences, combine them
            if meaningful_sentences:
                explanation = " ".join(meaningful_sentences)
        
        # If still no good explanation, use the full text
        if not explanation or len(explanation) < 50:
            # Remove any JSON-like structures
            cleaned_text = re.sub(r'{.*}', '', text, flags=re.DOTALL)
            explanation = cleaned_text
        
        # Clean up the explanation
        # Remove common JSON artifacts that might appear in the text
        explanation = re.sub(r'"[a-zA-Z_]+":', '', explanation)
        explanation = re.sub(r'["{}[\]]', '', explanation)
        
        # Limit length but ensure we have a substantial explanation
        if len(explanation) > 1000:
            explanation = explanation[:997] + "..."
            
        # Ensure explanation is meaningful
        if len(explanation.strip()) < 50:
            # Check if text indicates minimal content
            if any(phrase in text.lower() for phrase in ["insufficient content", "minimal content", "limited content"]):
                explanation = "This website appears to have insufficient content to make a reliable classification. The content is minimal, generic, or lacks industry-specific terminology that would allow for confident categorization."
            else:
                # If explanation is still too short, generate a basic one based on the classification
                explanation = "Classification based on analysis of website content. Unable to extract detailed reasoning from the analysis."
        
        return explanation
