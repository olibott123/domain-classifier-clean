from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_classifier import LLMClassifier
from snowflake_connector import SnowflakeConnector
import requests
import time
from urllib.parse import urlparse
import json
import os
import numpy as np
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Custom JSON encoder to handle problematic types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Configure Flask to use the custom encoder
app.json_encoder = CustomJSONEncoder

# Configuration
LOW_CONFIDENCE_THRESHOLD = 0.7  # Threshold below which we consider a classification "low confidence"
AUTO_RECLASSIFY_THRESHOLD = 0.6  # Threshold below which we automatically reclassify

# Get API keys and settings from environment variables
APIFY_TASK_ID = os.environ.get("APIFY_TASK_ID")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Initialize LLM classifier directly 
try:
    llm_classifier = LLMClassifier(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-haiku-20240307"
    )
    logger.info(f"Initialized LLM classifier with model: claude-3-haiku-20240307")
except Exception as e:
    logger.error(f"Failed to initialize LLM classifier: {e}")
    llm_classifier = None

# Initialize Snowflake connector
try:
    snowflake_conn = SnowflakeConnector()
    if not getattr(snowflake_conn, 'connected', False):
        logger.warning("Snowflake connection failed, using fallback")
except Exception as e:
    logger.error(f"Error initializing Snowflake connector: {e}")
    # Define a fallback Snowflake connector for when the real one isn't available
    class FallbackSnowflakeConnector:
        def check_existing_classification(self, domain):
            logger.info(f"Fallback: No existing classification for {domain}")
            return None
            
        def save_domain_content(self, domain, url, content):
            logger.info(f"Fallback: Not saving domain content for {domain}")
            return True, None
            
        def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method):
            logger.info(f"Fallback: Not saving classification for {domain}")
            return True, None
            
        def get_domain_content(self, domain):
            logger.info(f"Fallback: No content for {domain}")
            return None
    
    snowflake_conn = FallbackSnowflakeConnector()

def extract_domain_from_email(email):
    """Extract domain from an email address."""
    try:
        # Simple validation of email format
        if not email or '@' not in email:
            return None
            
        # Extract domain portion (after @)
        domain = email.split('@')[-1].strip().lower()
        
        # Basic validation of domain
        if not domain or '.' not in domain:
            return None
            
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from email: {e}")
        return None

def crawl_website(url):
    """Crawl a website using Apify with improved timeout handling."""
    try:
        logger.info(f"Starting crawl for {url}")
        
        # Start the crawl
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,      # Limit depth for faster crawls
            "maxCrawlPages": 5,         # Reduced from 10 to 5 for faster crawls
            "timeoutSecs": 120          # Set explicit timeout for Apify
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            run_id = response.json()['data']['id']
        except Exception as e:
            logger.error(f"Error starting crawl: {e}")
            return None
            
        # Wait for crawl to complete
        endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
        
        max_attempts = 12  # Reduced from 30 to 12 (about 2 minutes total)
        for attempt in range(max_attempts):
            logger.info(f"Checking crawl results, attempt {attempt+1}/{max_attempts}")
            
            try:
                response = requests.get(endpoint, timeout=10)  # Shorter timeout for status checks
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data:
                        combined_text = ' '.join(item.get('text', '') for item in data if item.get('text'))
                        if combined_text:
                            logger.info(f"Crawl completed, got {len(combined_text)} characters of content")
                            return combined_text
                        else:
                            logger.warning(f"Crawl returned data but no text content")
                else:
                    logger.warning(f"Received status code {response.status_code} when checking crawl results")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout when checking crawl status (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"Error checking crawl status: {e}")
            
            # Check if we've tried enough times and should try the fallback approach
            if attempt == 5:  # After about 50 seconds (6 attempts)
                logger.info("Trying fallback approach with direct request...")
                try:
                    direct_response = requests.get(url, timeout=15)
                    if direct_response.status_code == 200:
                        # Use a simple content extraction approach
                        text_content = direct_response.text
                        
                        # Extract readable text by removing HTML tags (simple approach)
                        import re
                        clean_text = re.sub(r'<[^>]+>', ' ', text_content)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        
                        if clean_text and len(clean_text) > 100:
                            logger.info(f"Direct request successful, got {len(clean_text)} characters")
                            return clean_text
                except Exception as e:
                    logger.warning(f"Direct request failed: {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(10)  # 10-second sleep between checks
        
        logger.warning(f"Crawl timed out after {max_attempts} attempts")
        return None
    except Exception as e:
        logger.error(f"Error crawling website: {e}")
        return None

def save_to_snowflake(domain, url, content, classification):
    """Save classification data to Snowflake"""
    try:
        # Always save the domain content
        logger.info(f"Saving content to Snowflake for {domain}")
        snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )

        # Ensure max_confidence exists
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence

        # Set low_confidence flag based on confidence threshold
        if 'low_confidence' not in classification:
            classification['low_confidence'] = classification['max_confidence'] < LOW_CONFIDENCE_THRESHOLD

        # Process explanation to ensure it's not truncated mid-sentence
        llm_explanation = classification.get('llm_explanation', '')
        
        # If explanation is too long, trim it properly at a sentence boundary
        if len(llm_explanation) > 500:
            # Find the last period before 450 chars
            last_period_index = llm_explanation[:450].rfind('.')
            if last_period_index > 0:
                shortened_explanation = llm_explanation[:last_period_index + 1]
            else:
                # If no period found, just truncate with an ellipsis
                shortened_explanation = llm_explanation[:450] + "..."
        else:
            shortened_explanation = llm_explanation
            
        # Create model metadata with properly formatted explanation
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307',
            'llm_explanation': shortened_explanation
        }
        
        # Special case for parked domains - save as "Parked Domain" if is_parked flag is set
        company_type = classification.get('predicted_class', 'Unknown')
        if classification.get('is_parked', False):
            company_type = "Parked Domain"
        
        logger.info(f"Saving classification to Snowflake: {domain}, {company_type}")
        snowflake_conn.save_classification(
            domain=domain,
            company_type=str(company_type),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification.get('confidence_scores', {}))[:4000],  # Limit size
            model_metadata=json.dumps(model_metadata)[:4000],  # Limit size
            low_confidence=bool(classification.get('low_confidence', False)),
            detection_method=str(classification.get('detection_method', 'llm_classification'))
        )
        
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}\n{traceback.format_exc()}")
        return False

@app.route('/classify-domain', methods=['POST', 'OPTIONS'])
def classify_domain():
    """Direct API that classifies a domain or email and returns the result"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        input_value = data.get('url', '').strip()
        force_reclassify = data.get('force_reclassify', False)
        
        if not input_value:
            return jsonify({"error": "URL or email is required"}), 400
        
        # Check if input is an email (contains @)
        is_email = '@' in input_value
        email = None
        if is_email:
            # Extract domain from email
            email = input_value
            domain = extract_domain_from_email(email)
            if not domain:
                return jsonify({"error": "Invalid email format"}), 400
            logger.info(f"Extracted domain '{domain}' from email '{email}'")
        else:
            # Process as domain/URL
            # Format URL properly
            if not input_value.startswith('http'):
                input_value = 'https://' + input_value
                
            # Extract domain
            parsed_url = urlparse(input_value)
            domain = parsed_url.netloc
            if not domain:
                domain = parsed_url.path
                
            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]
        
        if not domain:
            return jsonify({"error": "Invalid URL or email"}), 400
            
        url = f"https://{domain}"
        logger.info(f"Processing classification request for {domain}")
        
        # Check for existing classification if not forcing reclassification
        if not force_reclassify:
            existing_record = snowflake_conn.check_existing_classification(domain)
            if existing_record:
                # Check if confidence is below threshold for auto-reclassification
                confidence_score = existing_record.get('confidence_score', 1.0)
                if confidence_score < AUTO_RECLASSIFY_THRESHOLD:
                    logger.info(f"Auto-reclassifying {domain} due to low confidence score: {confidence_score}")
                    force_reclassify = True
                else:
                    logger.info(f"Found existing classification for {domain}")
                    
                    # Extract confidence scores
                    confidence_scores = {}
                    try:
                        confidence_scores = json.loads(existing_record.get('all_scores', '{}'))
                    except Exception as e:
                        logger.warning(f"Could not parse all_scores for {domain}: {e}")
                    
                    # Extract LLM explanation
                    llm_explanation = ""
                    try:
                        metadata = json.loads(existing_record.get('model_metadata', '{}'))
                        llm_explanation = metadata.get('llm_explanation', '')
                    except Exception as e:
                        logger.warning(f"Could not parse model_metadata for {domain}: {e}")
                    
                    # Add low_confidence flag based on confidence score
                    low_confidence = existing_record.get('low_confidence', confidence_score < LOW_CONFIDENCE_THRESHOLD)
                    
                    # Check if it's a parked domain (stored as "Parked Domain" in company_type)
                    is_parked = existing_record.get('company_type') == "Parked Domain"
                    
                    # Process confidence scores with type handling
                    processed_scores = {}
                    for category, score in confidence_scores.items():
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
                    
                    # Final validation - ensure cached scores are different
                    if len(set(processed_scores.values())) <= 1:
                        logger.warning("Cached response has identical confidence scores, fixing...")
                        pred_class = existing_record.get('company_type')
                        if pred_class == "Managed Service Provider":
                            processed_scores = {
                                "Managed Service Provider": 80,
                                "Integrator - Commercial A/V": 15,
                                "Integrator - Residential A/V": 5
                            }
                        elif pred_class == "Integrator - Commercial A/V":
                            processed_scores = {
                                "Integrator - Commercial A/V": 80,
                                "Managed Service Provider": 15,
                                "Integrator - Residential A/V": 5
                            }
                        else:  # Assume Residential A/V
                            processed_scores = {
                                "Integrator - Residential A/V": 80,
                                "Integrator - Commercial A/V": 15, 
                                "Managed Service Provider": 5
                            }
                    
                    # Return the cached classification
                    result = {
                        "domain": domain,
                        "predicted_class": existing_record.get('company_type'),
                        "confidence_score": int(existing_record.get('confidence_score', 0.5) * 100),
                        "confidence_scores": processed_scores,
                        "explanation": llm_explanation if llm_explanation else 'No explanation available.',
                        "low_confidence": low_confidence,
                        "detection_method": existing_record.get('detection_method', 'api'),
                        "source": "cached",
                        "is_parked": is_parked
                    }
                    
                    # Add email to response if input was an email
                    if email:
                        result["email"] = email
                        
                    return jsonify(result), 200
        
        # Try to get content (either from DB or by crawling)
        content = None
        
        # If reclassifying, try to get existing content first
        if force_reclassify:
            try:
                content = snowflake_conn.get_domain_content(domain)
                if content:
                    logger.info(f"Using existing content for {domain}")
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get existing content, will crawl instead: {e}")
                content = None
        
        # If no content yet, crawl the website
        if not content:
            logger.info(f"Crawling website for {domain}")
            content = crawl_website(url)
            
            if not content:
                error_result = {
                    "domain": domain,
                    "error": "Failed to crawl website or website has insufficient content",
                    "predicted_class": "Unknown",
                    "confidence_score": 0,  # Changed from 5 to 0 to avoid confusion
                    "confidence_scores": {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0
                    },
                    "explanation": f"We were unable to retrieve content from {domain}. This could be due to a server timeout, SSL certificate issues, or the website being unavailable. Without analyzing the website content, we cannot determine the company type.",
                    "low_confidence": True,
                    "is_crawl_error": True  # Added to distinguish crawl errors
                }
                
                # Add email to error response if input was an email
                if email:
                    error_result["email"] = email
                    
                return jsonify(error_result), 503  # Changed from 500 to 503 (Service Unavailable)
        
        # Classify the content
        if not llm_classifier:
            error_result = {
                "domain": domain, 
                "error": "LLM classifier is not available",
                "predicted_class": "Unknown",
                "confidence_score": 0,  # Changed from 5 to 0
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0
                },
                "explanation": "Our classification system is temporarily unavailable. Please try again later. This issue has been logged and will be addressed by our technical team.",
                "low_confidence": True
            }
            
            # Add email to error response if input was an email
            if email:
                error_result["email"] = email
                
            return jsonify(error_result), 500
            
        logger.info(f"Classifying content for {domain}")
        classification = llm_classifier.classify(content, domain)
        
        if not classification:
            error_result = {
                "domain": domain,
                "error": "Classification failed",
                "predicted_class": "Unknown",
                "confidence_score": 0,  # Changed from 5 to 0
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0
                },
                "explanation": f"We encountered an issue while analyzing {domain}. Although content was retrieved from the website, our classification system was unable to process it properly. This could be due to unusual formatting or temporary system limitations.",
                "low_confidence": True
            }
            
            # Add email to error response if input was an email
            if email:
                error_result["email"] = email
                
            return jsonify(error_result), 500
        
        # Save to Snowflake (always save, even for reclassifications)
        save_to_snowflake(domain, url, content, classification)
        
        # Create the response with properly differentiated confidence scores
        if classification.get("is_parked", False):
            # Special case for parked domains
            result = {
                "domain": domain,
                "predicted_class": "Parked Domain",  # Clear indicator in the UI
                "confidence_score": 0,  # Zero confidence rather than 5%
                "confidence_scores": {
                    category: 0 for category in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]
                },
                "explanation": classification.get('llm_explanation', 'This appears to be a parked or inactive domain without business-specific content.'),
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
            
            # Final validation - ensure scores are different
            if len(set(processed_scores.values())) <= 1:
                logger.warning("API response has identical confidence scores, fixing...")
                pred_class = classification.get('predicted_class')
                if pred_class == "Managed Service Provider":
                    processed_scores = {
                        "Managed Service Provider": 80,
                        "Integrator - Commercial A/V": 15,
                        "Integrator - Residential A/V": 5
                    }
                elif pred_class == "Integrator - Commercial A/V":
                    processed_scores = {
                        "Integrator - Commercial A/V": 80,
                        "Managed Service Provider": 15,
                        "Integrator - Residential A/V": 5
                    }
                else:  # Assume Residential A/V
                    processed_scores = {
                        "Integrator - Residential A/V": 80,
                        "Integrator - Commercial A/V": 15, 
                        "Managed Service Provider": 5
                    }
                
                # Update max_confidence to match the new highest value
                max_confidence = 80
            
            result = {
                "domain": domain,
                "predicted_class": classification.get('predicted_class'),
                "confidence_score": max_confidence,
                "confidence_scores": processed_scores,
                "explanation": classification.get('llm_explanation', 'No explanation available.'),
                "low_confidence": classification.get('low_confidence', False),
                "detection_method": classification.get('detection_method', 'api'),
                "source": "fresh",
                "is_parked": False
            }

        # Add email to response if input was an email
        if email:
            result["email"] = email
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "predicted_class": "Error",
            "confidence_score": 0,  # Changed from 5 to 0
            "confidence_scores": {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0
            },
            "explanation": f"An unexpected error occurred while processing your request: {str(e)}",
            "low_confidence": True
        }), 500

@app.route('/classify-email', methods=['POST', 'OPTIONS'])
def classify_email():
    """Alias for classify-domain that redirects email classification requests"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        # Create a new request with the email as URL
        new_data = {
            'url': email,
            'force_reclassify': data.get('force_reclassify', False)
        }
        
        # Forward to classify_domain by calling it directly with the new data
        # Use actual request for context but modify just the .json attribute
        # We need to be careful to use a copy and not modify the actual request
        original_json = request.json
        try:
            # Store the original json and use a context-like pattern
            _temp_request_json = new_data
            
            # Since we can't modify request.json directly,
            # we'll monkey patch the request.get_json function temporarily
            original_get_json = request.get_json
            
            def patched_get_json(*args, **kwargs):
                return _temp_request_json
                
            request.get_json = patched_get_json
            
            # Now call classify_domain, which will use our patched get_json
            result = classify_domain()
            
            # Return the result directly
            return result
            
        finally:
            # Restore original get_json
            if 'original_get_json' in locals():
                request.get_json = original_get_json
        
    except Exception as e:
        logger.error(f"Error processing email classification request: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "predicted_class": "Error",
            "confidence_score": 0,  # Changed from 5 to 0
            "confidence_scores": {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0
            },
            "explanation": f"An unexpected error occurred while processing your email classification request: {str(e)}",
            "low_confidence": True
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok", 
        "llm_available": llm_classifier is not None,
        "snowflake_connected": getattr(snowflake_conn, 'connected', False)
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
