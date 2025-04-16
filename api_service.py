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
import re
import socket

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
            
        def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method, llm_explanation):
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

def detect_error_type(error_message):
    """
    Analyze error message to determine the specific type of error.
    
    Args:
        error_message (str): The error message string
        
    Returns:
        tuple: (error_type, detailed_message)
    """
    error_message = str(error_message).lower()
    
    # SSL Certificate errors
    if any(phrase in error_message for phrase in ['certificate has expired', 'certificate verify failed', 'ssl', 'cert']):
        if 'expired' in error_message:
            return "ssl_expired", "The website's SSL certificate has expired."
        elif 'verify failed' in error_message:
            return "ssl_invalid", "The website has an invalid SSL certificate."
        else:
            return "ssl_error", "The website has SSL certificate issues."
    
    # DNS resolution errors
    elif any(phrase in error_message for phrase in ['getaddrinfo failed', 'name or service not known', 'no such host']):
        return "dns_error", "The domain could not be resolved. It may not exist or DNS records may be misconfigured."
    
    # Connection errors
    elif any(phrase in error_message for phrase in ['connection refused', 'connection timed out', 'connection error']):
        return "connection_error", "Could not establish a connection to the website. It may be down or blocking our requests."
    
    # 4XX HTTP errors
    elif any(phrase in error_message for phrase in ['403', 'forbidden', '401', 'unauthorized']):
        return "access_denied", "Access to the website was denied. The site may be blocking automated access."
    elif '404' in error_message or 'not found' in error_message:
        return "not_found", "The requested page was not found on this website."
    
    # 5XX HTTP errors
    elif any(phrase in error_message for phrase in ['500', '502', '503', '504', 'server error']):
        return "server_error", "The website is experiencing server errors."
    
    # Robots.txt or crawling restrictions
    elif any(phrase in error_message for phrase in ['robots.txt', 'disallowed', 'blocked by robots']):
        return "robots_restricted", "The website has restricted automated access in its robots.txt file."
    
    # Default fallback
    return "unknown_error", "An unknown error occurred while trying to access the website."

def crawl_website(url):
    """Crawl a website using Apify with improved timeout handling and error detection."""
    try:
        logger.info(f"Starting crawl for {url}")
        
        # Quick DNS check before attempting full crawl
        try:
            domain = urlparse(url).netloc
            socket.gethostbyname(domain)
        except socket.gaierror:
            logger.warning(f"Domain {domain} does not resolve - DNS error")
            return None, ("dns_error", "This domain does not exist or cannot be resolved")
        
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
            return None, detect_error_type(e)
            
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
                            return combined_text, (None, None)
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
                            return clean_text, (None, None)
                except Exception as e:
                    error_type, error_detail = detect_error_type(e)
                    logger.warning(f"Direct request failed: {e} (Type: {error_type})")
                    
                    # Don't return error here, keep trying the main Apify approach
            
            if attempt < max_attempts - 1:
                time.sleep(10)  # 10-second sleep between checks
        
        logger.warning(f"Crawl timed out after {max_attempts} attempts")
        return None, ("timeout", "The website took too long to respond. It may be experiencing performance issues.")
    except Exception as e:
        error_type, error_detail = detect_error_type(e)
        logger.error(f"Error crawling website: {e} (Type: {error_type})")
        return None, (error_type, error_detail)

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

        # Get explanation directly from classification
        llm_explanation = classification.get('llm_explanation', '')
        
        # If explanation is too long, trim it properly at a sentence boundary
        if len(llm_explanation) > 4000:
            # Find the last period before 3900 chars
            last_period_index = llm_explanation[:3900].rfind('.')
            if last_period_index > 0:
                llm_explanation = llm_explanation[:last_period_index + 1]
            else:
                # If no period found, just truncate with an ellipsis
                llm_explanation = llm_explanation[:3900] + "..."
            
        # Create model metadata
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307'
        }
        
        # Convert model metadata to JSON string
        model_metadata_json = json.dumps(model_metadata)[:4000]  # Limit size
            
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
            model_metadata=model_metadata_json,
            low_confidence=bool(classification.get('low_confidence', False)),
            detection_method=str(classification.get('detection_method', 'llm_classification')),
            llm_explanation=llm_explanation  # Add explanation directly to save_classification
        )
        
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}\n{traceback.format_exc()}")
        return False

def create_error_result(domain, error_type=None, error_detail=None, email=None):
    """
    Create a standardized error response based on the error type.
    
    Args:
        domain (str): The domain being processed
        error_type (str): The type of error detected
        error_detail (str): Detailed explanation of the error
        email (str, optional): Email address if processing an email
        
    Returns:
        dict: Standardized error response
    """
    # Default error response
    error_result = {
        "domain": domain,
        "error": "Failed to crawl website",
        "predicted_class": "Unknown",
        "confidence_score": 0,
        "confidence_scores": {
            "Managed Service Provider": 0,
            "Integrator - Commercial A/V": 0,
            "Integrator - Residential A/V": 0
        },
        "low_confidence": True,
        "is_crawl_error": True
    }
    
    # Add email if provided
    if email:
        error_result["email"] = email
    
    # Default explanation
    explanation = f"We were unable to retrieve content from {domain}. This could be due to a server timeout or the website being unavailable. Without analyzing the website content, we cannot determine the company type."
    
    # Enhanced error handling based on error type
    if error_type:
        error_result["error_type"] = error_type
        
        if error_type.startswith('ssl_'):
            explanation = f"We couldn't analyze {domain} because of SSL certificate issues. "
            if error_type == 'ssl_expired':
                explanation += f"The website's SSL certificate has expired. This is a security issue with the target website, not our classification service."
            elif error_type == 'ssl_invalid':
                explanation += f"The website has an invalid SSL certificate. This is a security issue with the target website, not our classification service."
            else:
                explanation += f"This is a security issue with the target website, not our classification service."
            
            error_result["is_ssl_error"] = True
            
        elif error_type == 'dns_error':
            explanation = f"We couldn't analyze {domain} because the domain could not be resolved. This typically means the domain doesn't exist or its DNS records are misconfigured."
            error_result["is_dns_error"] = True
            
        elif error_type == 'connection_error':
            explanation = f"We couldn't analyze {domain} because a connection couldn't be established. The website may be down, temporarily unavailable, or blocking our requests."
            error_result["is_connection_error"] = True
            
        elif error_type == 'access_denied':
            explanation = f"We couldn't analyze {domain} because access was denied. The website may be blocking automated access or requiring authentication."
            error_result["is_access_denied"] = True
            
        elif error_type == 'not_found':
            explanation = f"We couldn't analyze {domain} because the main page was not found. The website may be under construction or have moved to a different URL."
            error_result["is_not_found"] = True
            
        elif error_type == 'server_error':
            explanation = f"We couldn't analyze {domain} because the website is experiencing server errors. This is an issue with the target website, not our classification service."
            error_result["is_server_error"] = True
            
        elif error_type == 'robots_restricted':
            explanation = f"We couldn't analyze {domain} because the website restricts automated access. This is a policy set by the website owner."
            error_result["is_robots_restricted"] = True
            
        elif error_type == 'timeout':
            explanation = f"We couldn't analyze {domain} because the website took too long to respond. The website may be experiencing performance issues or temporarily unavailable."
            error_result["is_timeout"] = True
            
        # If we have a specific error detail, use it to enhance the explanation
        if error_detail:
            explanation += f" {error_detail}"
    
    error_result["explanation"] = explanation
    return error_result

def validate_result_consistency(result, domain):
    """
    Validate and ensure consistency between predicted_class, confidence scores, and explanation.
    
    Args:
        result (dict): The classification result
        domain (str): The domain being processed
        
    Returns:
        dict: The validated and consistent result
    """
    if not result:
        return result
        
    # First, ensure predicted_class is never null
    if result.get("predicted_class") is None:
        # Extract a class from the explanation if possible
        explanation = result.get("explanation", "")
        if "non-service business" in explanation.lower():
            result["predicted_class"] = "Non-Service Business"
        elif "vacation rental" in explanation.lower() or "travel" in explanation.lower():
            result["predicted_class"] = "Non-Service Business"
        elif "parked domain" in explanation.lower():
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
                logger.warning(f"Correcting predicted_class from {result['predicted_class']} to Non-Service Business based on explanation")
                result["predicted_class"] = "Non-Service Business"
                
    # Ensure "Process Did Not Complete" has 0% confidence
    if result.get("predicted_class") == "Process Did Not Complete":
        result["confidence_score"] = 0
        if "confidence_scores" in result:
            result["confidence_scores"] = {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0
            }
    
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
                    new_explanation += f"STEP 5: Since this is classified as a service business, there is no need to assess the internal IT potential\n\n"
                else:
                    # Try to extract internal IT potential from confidence scores
                    it_potential = 50  # Default
                    if "confidence_scores" in result and "Corporate IT" in result["confidence_scores"]:
                        it_potential = result["confidence_scores"]["Corporate IT"]
                    
                    new_explanation += f"STEP 3: The company is NOT a service/management business that provides ongoing IT or A/V services to clients\n\n"
                    new_explanation += f"STEP 4: Since this is not a service business, we classify it as {predicted_class}\n\n"
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
    
    return result

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
        use_existing_content = data.get('use_existing_content', False)
        
        # Always set force_reclassify to true for new requests unless explicitly using existing content
        if request.method == 'POST' and 'use_existing_content' not in data:
            # Set a default preference for fresh classification
            force_reclassify = data.get('force_reclassify', True)
        
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
                # Log the full record for debugging
                logger.info(f"Retrieved record from Snowflake: {existing_record}")
                
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
                    
                    # Extract LLM explanation directly from the LLM_EXPLANATION column
                    llm_explanation = existing_record.get('LLM_EXPLANATION', '')
                    
                    # If LLM_EXPLANATION is not available, try to get it from model_metadata
                    if not llm_explanation:
                        try:
                            metadata = json.loads(existing_record.get('model_metadata', '{}'))
                            llm_explanation = metadata.get('llm_explanation', '')
                        except Exception as e:
                            logger.warning(f"Could not parse model_metadata for {domain}: {e}")
                    
                    # Ensure we have an explanation
                    if not llm_explanation:
                        llm_explanation = f"The domain {domain} was previously classified as a {existing_record.get('company_type')} based on analysis of website content."
                    
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
                    
                    # IMPORTANT - Only fix identical scores if all values are exactly the same
                    # This prevents overriding valid confidence distributions
                    identical_values = len(set(processed_scores.values())) <= 1
                    all_values_zero = all(value == 0 for value in processed_scores.values())

                    if identical_values and not all_values_zero:
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
                        elif pred_class == "Integrator - Residential A/V":
                            processed_scores = {
                                "Integrator - Residential A/V": 80,
                                "Integrator - Commercial A/V": 15, 
                                "Managed Service Provider": 5
                            }
                        elif pred_class == "Non-Service Business":
                            # For non-service business, add Corporate IT score
                            internal_it_potential = existing_record.get('internal_it_potential', 50)
                            if internal_it_potential is None:
                                internal_it_potential = 50
                                
                            processed_scores = {
                                "Managed Service Provider": 5,
                                "Integrator - Commercial A/V": 3,
                                "Integrator - Residential A/V": 2,
                                "Corporate IT": internal_it_potential  # Add Corporate IT score
                            }
                            
                    # Add Corporate IT score for Non-Service Business classifications if missing
                    if existing_record.get('company_type') == "Non-Service Business" and "Corporate IT" not in processed_scores:
                        # Try to extract internal IT potential from explanation or set a default
                        it_potential = 50
                        it_match = re.search(r'internal IT.*?(\d+)[/\s]*100', llm_explanation)
                        if it_match:
                            it_potential = int(it_match.group(1))
                            
                        processed_scores["Corporate IT"] = it_potential
                        # Ensure service scores are low
                        for category in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                            processed_scores[category] = min(processed_scores.get(category, 5), 10)
                    
                    # Return the cached classification
                    result = {
                        "domain": domain,
                        "predicted_class": existing_record.get('company_type'),
                        "confidence_score": int(existing_record.get('confidence_score', 0.5) * 100),
                        "confidence_scores": processed_scores,
                        "explanation": llm_explanation,  # Include the explanation here
                        "low_confidence": low_confidence,
                        "detection_method": existing_record.get('detection_method', 'api'),
                        "source": "cached",
                        "is_parked": is_parked
                    }
                    
                    # Add email to response if input was an email
                    if email:
                        result["email"] = email
                    
                    # Ensure result consistency
                    result = validate_result_consistency(result, domain)
                    
                    # Log the response for debugging
                    logger.info(f"Sending response to client: {json.dumps(result)}")
                        
                    return jsonify(result), 200
        
        # Try to get content (either from DB or by crawling)
        content = None
        
        # If reclassifying or using existing content, try to get existing content first
        if force_reclassify or use_existing_content:
            try:
                content = snowflake_conn.get_domain_content(domain)
                if content:
                    logger.info(f"Using existing content for {domain}")
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get existing content, will crawl instead: {e}")
                content = None

        # If we specifically requested to use existing content but none was found
        if use_existing_content and not content:
            error_result = {
                "domain": domain,
                "error": "No existing content found",
                "predicted_class": "Unknown",
                "confidence_score": 0,
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0
                },
                "explanation": f"We could not find previously stored content for {domain}. Please try recrawling instead.",
                "low_confidence": True,
                "no_existing_content": True
            }
            
            # Add email to response if input was an email
            if email:
                error_result["email"] = email
                
            return jsonify(error_result), 404
        
        # If no content yet and we're not using existing content or existing content wasn't found, crawl the website
        error_type = None
        error_detail = None
        
        if not content and not use_existing_content:
            logger.info(f"Crawling website for {domain}")
            content, (error_type, error_detail) = crawl_website(url)
            
            if not content:
                error_result = create_error_result(domain, error_type, error_detail, email)
                return jsonify(error_result), 503  # Service Unavailable
        
        # Classify the content
        if not llm_classifier:
            error_result = {
                "domain": domain, 
                "error": "LLM classifier is not available",
                "predicted_class": "Unknown",
                "confidence_score": 0,
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
                "confidence_score": 0,
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
                elif pred_class == "Integrator - Residential A/V":  # Residential A/V
                    processed_scores = {
                        "Integrator - Residential A/V": 80,
                        "Integrator - Commercial A/V": 15, 
                        "Managed Service Provider": 5
                    }
                elif pred_class == "Process Did Not Complete":
                    # Set all scores to 0 for process_did_not_complete
                    processed_scores = {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0
                    }
                    # Reset max_confidence to 0.0
                    max_confidence = 0
                elif pred_class == "Non-Service Business":
                    # For non-service business, add Corporate IT score
                    internal_it_potential = classification.get('internal_it_potential', 50)
                    if internal_it_potential is None:
                        internal_it_potential = 50
                        
                    processed_scores = {
                        "Managed Service Provider": 5,
                        "Integrator - Commercial A/V": 3,
                        "Integrator - Residential A/V": 2,
                        "Corporate IT": internal_it_potential  # Add Corporate IT score
                    }
                
                # Update max_confidence to match the new highest value if not Process Did Not Complete
                if pred_class not in ["Process Did Not Complete", "Non-Service Business"]:
                    max_confidence = 80
                    
            # Ensure explanation exists
            explanation = classification.get('llm_explanation', '')
            if not explanation:
                explanation = f"Based on analysis of website content, {domain} has been classified as a {classification.get('predicted_class')}."
                
            # Check for Non-Service Business in the explanation 
            if "non-service business" in explanation.lower() and classification.get('predicted_class') in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                if max_confidence <= 20:  # Only override if confidence is low
                    logger.info(f"Correcting classification for {domain} to Non-Service Business based on explanation")
                    classification['predicted_class'] = "Non-Service Business"

            # Add Corporate IT for Non-Service Business if not already present
            if classification.get('predicted_class') == "Non-Service Business" and "Corporate IT" not in processed_scores:
                internal_it_potential = classification.get('internal_it_potential', 50)
                if internal_it_potential is None:
                    internal_it_potential = 50
                    
                processed_scores["Corporate IT"] = internal_it_potential
                # Ensure service scores are low
                for category in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]:
                    processed_scores[category] = min(processed_scores.get(category, 5), 10)
            
            result = {
                "domain": domain,
                "predicted_class": classification.get('predicted_class'),
                "confidence_score": max_confidence,
                "confidence_scores": processed_scores,
                "explanation": explanation,  # Include the explanation here
                "low_confidence": classification.get('low_confidence', False),
                "detection_method": classification.get('detection_method', 'api'),
                "source": "fresh",
                "is_parked": False
            }

        # Add email to response if input was an email
        if email:
            result["email"] = email
            
        # Ensure result consistency
        result = validate_result_consistency(result, domain)
        
        # Log the response for debugging
        logger.info(f"Sending response to client: {json.dumps(result)}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        # Try to identify the error type if possible
        error_type, error_detail = detect_error_type(str(e))
        error_result = create_error_result(domain if 'domain' in locals() else "unknown", error_type, error_detail, email if 'email' in locals() else None)
        error_result["error"] = str(e)  # Add the actual error message
        return jsonify(error_result), 500

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
            'force_reclassify': data.get('force_reclassify', False),
            'use_existing_content': data.get('use_existing_content', False)
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
        error_type, error_detail = detect_error_type(str(e))
        error_result = create_error_result("unknown", error_type, error_detail)
        error_result["error"] = str(e)
        return jsonify(error_result), 500

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
