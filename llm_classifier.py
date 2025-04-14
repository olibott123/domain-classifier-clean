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
                    "Integrator - Commercial A/V": 0.05,
                    "Integrator - Residential A/V": 0.05,
                    "Managed Service Provider": 0.05
                },
                "max_confidence": 0.05,
                "llm_explanation": "This domain appears to be parked or has minimal content. It may be a placeholder, under construction, or for sale. There is insufficient information to determine the company type.",
                "detection_method": "minimal_content_detection"
            }
            
        try:
            # Prepare the input for Claude
            domain_context = f" for the website {domain}" if domain else ""
            
            # Define the response format we want from Claude
            response_format = {
                "predicted_class": "The most likely category (one of: 'Integrator - Commercial A/V', 'Integrator - Residential A/V', 'Managed Service Provider')",
                "confidence_scores": {
                    "Integrator - Commercial A/V": "float between 0 and 1",
                    "Integrator - Residential A/V": "float between 0 and 1",
                    "Managed Service Provider": "float between 0 and 1"
                },
                "llm_explanation": "A detailed explanation of why this classification was chosen, citing specific evidence from the text"
            }
            
            # Generate a prompt that asks for detailed reasoning and structured output
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

Analyze the provided website content carefully. If there is insufficient content to make a classification, state this explicitly and assign low confidence scores.

VERY IMPORTANT: Your explanation MUST be detailed (at least 3-4 sentences) and reference SPECIFIC evidence from the text that supports your classification. Generic explanations are not acceptable.

Also CRITICAL: If the website content is minimal, generic, or doesn't clearly indicate the company's business, assign LOW confidence scores (below 0.3) to ALL categories and make sure your explanation notes the lack of clear evidence.

You MUST provide DIFFERENT confidence scores for each category - they should NOT all be the same value. The category with the most evidence should have the highest confidence score.

You MUST provide your analysis in JSON format with the following structure:
{json.dumps(response_format, indent=2)}
"""

            user_prompt = f"Based on the following website content{domain_context}, classify the company and explain your reasoning in detail. Focus on specific evidence from the text that supports your conclusion. If the content is minimal or generic, explicitly state this and assign low confidence scores. Your explanation must cite specific phrases or terms from the website content:\n\n{text[:15000]}"

            request_data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1500,  # Increased to allow for longer explanations
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
                                json_str = re.search(r'({.*"predicted_class".*})', text_content, re.DOTALL)
                                
                                if not json_str:
                                    logger.warning("Could not find JSON in LLM response, falling back to text parsing")
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
                                else:
                                    # Check if all confidence scores are the same
                                    scores = list(parsed_json["confidence_scores"].values())
                                    if len(scores) > 1 and all(score == scores[0] for score in scores):
                                        # If all scores are the same, regenerate differentiated scores
                                        logger.warning("All confidence scores are the same, regenerating differentiated scores")
                                        parsed_json["confidence_scores"] = self._extract_confidence_scores(text_content)
                                
                                # Ensure llm_explanation is included and has substance
                                if "llm_explanation" not in parsed_json or not parsed_json["llm_explanation"] or len(parsed_json["llm_explanation"].strip()) < 50:
                                    # Extract reasoning from the full text
                                    parsed_json["llm_explanation"] = self._extract_explanation(text_content)
                                    # If still too short, flag it
                                    if len(parsed_json["llm_explanation"].strip()) < 50:
                                        logger.warning("Explanation is too short, may not be useful")
                                
                                # Ensure predicted_class matches highest confidence score
                                confidence_scores = parsed_json.get("confidence_scores", {})
                                if confidence_scores:
                                    max_class = max(confidence_scores.items(), key=lambda x: x[1])
                                    parsed_json["predicted_class"] = max_class[0]
                                    parsed_json["max_confidence"] = max_class[1]
                                else:
                                    parsed_json["max_confidence"] = 0.7  # Default confidence
                                    
                                # Set detection method
                                parsed_json["detection_method"] = "llm_classification"
                                
                                # Log the explanation length for debugging
                                logger.info(f"Explanation length: {len(parsed_json.get('llm_explanation', ''))}")
                                
                                # Verify the explanation doesn't have the generic message
                                if "Classification based on analysis of website content" in parsed_json.get("llm_explanation", ""):
                                    logger.warning("Generic explanation detected, trying to extract better explanation")
                                    better_explanation = self._try_extract_better_explanation(text_content)
                                    if better_explanation:
                                        parsed_json["llm_explanation"] = better_explanation
                                
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
    
    def _try_extract_better_explanation(self, text: str) -> str:
        """Try to extract a better explanation from the full response text."""
        # Look for sentences that might contain reasoning
        sentences = re.split(r'(?<=[.!?])\s+', text)
        reasoning_sentences = []
        
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
                reasoning_sentences.append(sentence)
        
        # If we found some reasoning sentences, combine them
        if reasoning_sentences:
            explanation = " ".join(reasoning_sentences)
            if len(explanation) > 50:  # Ensure it's substantial
                return explanation
        
        # If we can't extract a better explanation, return None
        return None
    
    def _parse_free_text(self, text: str) -> Dict[str, Any]:
        """Parse classification from free-form text response."""
        result = {
            "confidence_scores": {
                "Integrator - Commercial A/V": 0.0,
                "Integrator - Residential A/V": 0.0,
                "Managed Service Provider": 0.0
            },
            "llm_explanation": ""
        }
        
        # First, examine the explanation to identify the main classification
        lower_text = text.lower()
        
        # Check for indications of minimal content
        minimal_content_phrases = [
            "not enough information", "insufficient information", "insufficient content",
            "minimal content", "limited content", "not enough content", "lack of content",
            "generic content", "generic landing page", "placeholder", "parked domain"
        ]
        
        if any(phrase in lower_text for phrase in minimal_content_phrases):
            logger.info("LLM response indicates minimal content")
            result["predicted_class"] = "Unknown"
            result["confidence_scores"]["Integrator - Commercial A/V"] = 0.05
            result["confidence_scores"]["Integrator - Residential A/V"] = 0.05
            result["confidence_scores"]["Managed Service Provider"] = 0.05
            result["max_confidence"] = 0.05
            result["llm_explanation"] = "There is insufficient content on this website to determine the company type. The website may be a generic landing page, parked domain, or under construction."
            result["detection_method"] = "minimal_content_detection"
            return result
        
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
        
        # Calculate confidence scores - ensure they're differentiated
        # Set base values first
        base_scores = {
            "Managed Service Provider": 0.3 if msp_score > 0 else 0.1,
            "Integrator - Commercial A/V": 0.3 if commercial_score > 0 else 0.1,
            "Integrator - Residential A/V": 0.3 if residential_score > 0 else 0.1
        }
        
        # Add weights based on indicators found
        total_indicators = msp_score + commercial_score + residential_score
        if total_indicators > 0:
            if msp_score > 0:
                base_scores["Managed Service Provider"] += 0.5 * (msp_score / total_indicators)
            if commercial_score > 0:
                base_scores["Integrator - Commercial A/V"] += 0.5 * (commercial_score / total_indicators)
            if residential_score > 0:
                base_scores["Integrator - Residential A/V"] += 0.5 * (residential_score / total_indicators)
        
        # Ensure the scores are different even if indicators are the same
        # If they would be all the same, slightly adjust them
        if (msp_score == commercial_score == residential_score) and msp_score > 0:
            # If we have a tie with indicators, look for additional signals
            if "server" in lower_text or "network" in lower_text or "cloud" in lower_text:
                base_scores["Managed Service Provider"] += 0.05
            if "conference" in lower_text or "business" in lower_text or "corporate" in lower_text:
                base_scores["Integrator - Commercial A/V"] += 0.03
            if "home" in lower_text or "theater" in lower_text or "residential" in lower_text:
                base_scores["Integrator - Residential A/V"] += 0.04
        
        # Set the final scores
        result["confidence_scores"] = base_scores
        
        # Set the predicted class
        max_class = max(result["confidence_scores"].items(), key=lambda x: x[1])
        result["predicted_class"] = max_class[0]
        result["max_confidence"] = max_class[1]
        
        # Extract explanation - try to get a substantial one
        result["llm_explanation"] = self._extract_explanation(text)
        
        return result
    
    def _extract_confidence_scores(self, text: str) -> Dict[str, float]:
        """Extract confidence scores from text response with differentiation."""
        scores = {
            "Integrator - Commercial A/V": 0.0,
            "Integrator - Residential A/V": 0.0,
            "Managed Service Provider": 0.0
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
                "Integrator - Commercial A/V": 0.05,
                "Integrator - Residential A/V": 0.05,
                "Managed Service Provider": 0.05
            }
            return scores
        
        # Look for explicit confidence values
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
        
        # Check if no scores were extracted or if all scores are the same
        all_scores_same = len(set(scores.values())) <= 1
        no_scores = all(score == 0.0 for score in scores.values())
        
        if no_scores or all_scores_same:
            # Analyze the text content to determine confidence scores
            lower_text = text.lower()
            
            # Count keyword mentions to differentiate scores
            msp_keywords = ["managed service", "msp", "it service", "it support", "network", "server", 
                          "cybersecurity", "helpdesk", "cloud service", "technical support"]
            commercial_keywords = ["commercial", "business", "corporate", "conference room", 
                                 "digital signage", "presentation", "enterprise", "office"]
            residential_keywords = ["residential", "home theater", "home automation", "smart home", 
                                  "living space", "whole-home", "luxury", "homeowner"]
            
            msp_count = sum(lower_text.count(keyword) for keyword in msp_keywords)
            commercial_count = sum(lower_text.count(keyword) for keyword in commercial_keywords)
            residential_count = sum(lower_text.count(keyword) for keyword in residential_keywords)
            
            # Calculate base scores (ensuring they are different)
            base_total = msp_count + commercial_count + residential_count
            if base_total > 0:
                # Start with a base confidence and add weighted counts
                scores["Managed Service Provider"] = 0.3 + (0.6 * msp_count / base_total)
                scores["Integrator - Commercial A/V"] = 0.25 + (0.6 * commercial_count / base_total)
                scores["Integrator - Residential A/V"] = 0.2 + (0.6 * residential_count / base_total)
            else:
                # If no keywords found, assign slightly different base scores
                scores["Managed Service Provider"] = 0.35
                scores["Integrator - Commercial A/V"] = 0.33
                scores["Integrator - Residential A/V"] = 0.32
        
        # Final check to ensure scores are different
        if len(set(scores.values())) <= 1:
            # Force differentiation with small adjustments
            scores["Managed Service Provider"] += 0.03
            scores["Integrator - Commercial A/V"] += 0.01
        
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
                    explanation = parts[1].strip()
                    break
        
        # If no marker found, try extracting sentences with reasoning keywords
        if not explanation or len(explanation) < 50:
            reasoned_explanation = self._try_extract_better_explanation(text)
            if reasoned_explanation:
                explanation = reasoned_explanation
        
        # If still no good explanation, use the full text
        if not explanation or len(explanation) < 50:
            explanation = text
        
        # Clean up the explanation
        # Remove any JSON-like formatting
        explanation = re.sub(r'{.*}', '', explanation, flags=re.DOTALL)
        
        # Check for meaningfulness
        if len(explanation.strip()) < 50 or "Classification based on analysis of website content" in explanation:
            # Find sentences with reasoning language
            sentences = re.split(r'(?<=[.!?])\s+', text)
            meaningful_sentences = []
            
            reasoning_terms = ["because", "since", "reason", "evidence", "indicates", "suggests", 
                             "mentions", "refers", "includes", "describes", "shows", "offers"]
            
            for sentence in sentences:
                if any(term in sentence.lower() for term in reasoning_terms):
                    meaningful_sentences.append(sentence)
            
            if meaningful_sentences:
                explanation = " ".join(meaningful_sentences)
        
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
                explanation = "Classification based on analysis of website content. Unable to extract specific reasoning from the analysis."
        
        return explanation
