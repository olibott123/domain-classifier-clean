"""Classification API routes for domain classifier."""
import logging
import traceback
from flask import request, jsonify
from urllib.parse import urlparse
from typing import Dict, Any, Tuple, Optional

# Import domain utilities
from domain_classifier.utils.domain_utils import extract_domain_from_email
from domain_classifier.utils.error_handling import detect_error_type, create_error_result, check_domain_dns, is_domain_worth_crawling

# Import configuration
from domain_classifier.config.overrides import check_domain_override

# Import services
from domain_classifier.crawlers.apify_crawler import crawl_website
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.classifiers.result_validator import validate_result_consistency
from domain_classifier.storage.cache_manager import process_cached_result
from domain_classifier.utils.text_processing import extract_company_description
from domain_classifier.storage.result_processor import process_fresh_result
from domain_classifier.classifiers.decision_tree import is_parked_domain, create_parked_domain_result

# Set up logging
logger = logging.getLogger(__name__)

def check_for_parked_domain(domain: str, url: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Perform a quick check to determine if a domain is parked before full crawling.
    
    Args:
        domain (str): The domain name
        url (str): The full URL
        
    Returns:
        tuple: (is_parked, result)
            - is_parked: Whether the domain is parked
            - result: The result dict if parked, None otherwise
    """
    try:
        from domain_classifier.crawlers.direct_crawler import direct_crawl
        from domain_classifier.classifiers.decision_tree import is_parked_domain, create_parked_domain_result
        
        logger.info(f"Performing quick check for parked domain: {domain}")
        quick_check_content, (error_type, error_detail), quick_crawler_type = direct_crawl(url, timeout=5.0)
        
        # Check if this is a parked domain
        if error_type == "is_parked" or (quick_check_content and is_parked_domain(quick_check_content, domain)):
            logger.info(f"Quick check detected parked domain: {domain}")
            parked_result = create_parked_domain_result(domain, crawler_type="quick_check_parked")
            
            # Process the result from the decision tree
            result = {
                "domain": domain,
                "predicted_class": "Parked Domain",
                "confidence_score": 0,
                "confidence_scores": {
                    "Managed Service Provider": 0,
                    "Integrator - Commercial A/V": 0,
                    "Integrator - Residential A/V": 0,
                    "Internal IT Department": 0
                },
                "explanation": parked_result.get("llm_explanation", f"The domain {domain} appears to be parked or inactive. This domain may be registered but not actively in use for a business."),
                "low_confidence": True,
                "is_parked": True,
                "final_classification": "6-Parked Domain - no enrichment",
                "crawler_type": "quick_check_parked",
                "classifier_type": "early_detection",
                "detection_method": "parked_domain_detection",
                "source": "fresh"
            }
            
            return True, result
            
        return False, None
    
    except Exception as e:
        logger.warning(f"Early parked domain check failed: {e}")
        return False, None

def register_classify_routes(app, llm_classifier, snowflake_conn):
    """Register domain/email classification related routes."""
    
    @app.route('/classify-domain', methods=['POST', 'OPTIONS'])
    def classify_domain():
        """Direct API that classifies a domain or email and returns the result"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            input_value = data.get('url', '').strip()
            
            # Change default for force_reclassify to True
            force_reclassify = data.get('force_reclassify', True)
            
            use_existing_content = data.get('use_existing_content', False)
            
            if not input_value:
                return jsonify({"error": "URL or email is required"}), 400
            
            # Check if input is an email
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
            
            # Check for domain override before any other processing
            domain_override = check_domain_override(domain)
            if domain_override:
                # Add email to response if input was an email
                if email:
                    domain_override["email"] = email
                    
                # Add website URL for clickable link
                domain_override["website_url"] = url
                
                # Add final classification based on predicted class
                if domain_override.get("predicted_class") == "Managed Service Provider":
                    domain_override["final_classification"] = "1-MSP"
                elif domain_override.get("predicted_class") == "Integrator - Commercial A/V":
                    domain_override["final_classification"] = "3-Commercial Integrator"
                elif domain_override.get("predicted_class") == "Integrator - Residential A/V":
                    domain_override["final_classification"] = "4-Residential Integrator"
                else:
                    domain_override["final_classification"] = "2-Internal IT"
                
                # Add crawler_type and classifier_type to ensure they appear at the bottom
                domain_override["crawler_type"] = "override"
                domain_override["classifier_type"] = "override"
                
                # Return the override directly
                logger.info(f"Sending override response to client: {domain_override}")
                return jsonify(domain_override), 200
            
            # Enhanced domain screening before attempting crawl
            worth_crawling, has_dns, dns_error, potentially_flaky = is_domain_worth_crawling(domain)
            
            # Initialize crawler_type
            crawler_type = None
            
            # Check for parked domain early
            if dns_error == "parked_domain":
                logger.info(f"Domain {domain} detected as parked domain during initial DNS check")
                parked_result = create_parked_domain_result(domain, crawler_type="dns_check_parked")
                result = process_fresh_result(parked_result, domain, email, url)
                result["crawler_type"] = "dns_check_parked"
                result["classifier_type"] = "early_detection"
                return jsonify(result), 200
            
            if not worth_crawling:
                logger.warning(f"Domain {domain} is not worth crawling: {dns_error}")
                error_result = create_error_result(domain, "dns_error" if "DNS" in dns_error else "connection_error", 
                                                 dns_error, email, crawler_type)
                error_result["website_url"] = url
                error_result["final_classification"] = "7-No Website available"
                return jsonify(error_result), 503  # Service Unavailable
            
            # If domain is worth crawling, do an additional early check for parked domains
            if worth_crawling:
                # Add early parked domain detection
                is_parked, parked_result = check_for_parked_domain(domain, url)
                if is_parked:
                    # Create result for the API
                    from domain_classifier.classifiers.decision_tree import create_parked_domain_result
                    parked_domain_data = create_parked_domain_result(domain, crawler_type="quick_check_parked")
                    
                    # Process the result
                    result = process_fresh_result(parked_domain_data, domain, email, url)
                    
                    # Add to final result
                    result["crawler_type"] = "quick_check_parked"  
                    result["classifier_type"] = "early_detection"
                    
                    return jsonify(result), 200
            
            if potentially_flaky:
                logger.warning(f"Domain {domain} passed basic checks but shows signs of being flaky (resetting connections)")
                # We'll still try to crawl, but warn the user that it might be unreliable
            else:
                logger.info(f"DNS check passed for domain: {domain}")
            
            # Check for existing classification if not forcing reclassification
            if not force_reclassify:
                existing_record = snowflake_conn.check_existing_classification(domain)
                
                if existing_record:
                    logger.info(f"Found existing classification for {domain}")
                    
                    # Process and return the cached result
                    result = process_cached_result(existing_record, domain, email, url)
                    
                    # Ensure result consistency
                    result = validate_result_consistency(result, domain)
                    
                    # Log the response for debugging
                    logger.info(f"Sending cached response to client")
                    
                    return jsonify(result), 200
            
            # Try to get content (either from DB or by crawling)
            content = None
            
            # If reclassifying or using existing content, try to get existing content first
            if force_reclassify or use_existing_content:
                try:
                    content = snowflake_conn.get_domain_content(domain)
                    if content:
                        logger.info(f"Using existing content for {domain}")
                        # Set crawler_type for existing content
                        crawler_type = "existing_content"
                        
                        # Add check for parked domain in cached content
                        from domain_classifier.classifiers.decision_tree import is_parked_domain
                        if is_parked_domain(content, domain):
                            logger.info(f"Detected parked domain from cached content: {domain}")
                            from domain_classifier.classifiers.decision_tree import create_parked_domain_result
                            parked_result = create_parked_domain_result(domain, crawler_type="cached_content_parked")
                            
                            # Process the parked domain result
                            result = process_fresh_result(parked_result, domain, email, url)
                            
                            # Add crawler_type and classifier_type to ensure they appear at the bottom
                            result["crawler_type"] = "cached_content_parked"
                            result["classifier_type"] = "early_detection"
                            
                            return jsonify(result), 200
                except Exception as e:
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
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": f"We could not find previously stored content for {domain}. Please try recrawling instead.",
                    "low_confidence": True,
                    "no_existing_content": True,
                    "website_url": url,
                    "final_classification": "2-Internal IT",
                    "crawler_type": "error_handler",
                    "classifier_type": "error_handler"
                }
                
                # Add email to response if input was an email
                if email:
                    error_result["email"] = email
                    
                return jsonify(error_result), 404
            
            # If no content yet and we're not using existing content, crawl the website
            error_type = None
            error_detail = None
            
            if not content and not use_existing_content:
                logger.info(f"Crawling website for {domain}")
                content, (error_type, error_detail), crawler_type = crawl_website(url)
                
                if not content:
                    error_result = create_error_result(domain, error_type, error_detail, email, crawler_type)
                    error_result["website_url"] = url
                    
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
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": "Our classification system is temporarily unavailable. Please try again later.",
                    "low_confidence": True,
                    "website_url": url,
                    "final_classification": "2-Internal IT",
                    "crawler_type": crawler_type or "error_handler",
                    "classifier_type": "error_handler"
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
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": f"We encountered an issue while analyzing {domain}.",
                    "low_confidence": True,
                    "website_url": url,
                    "final_classification": "2-Internal IT",
                    "crawler_type": crawler_type or "unknown",
                    "classifier_type": "error_handler"
                }
                
                # Add email to error response if input was an email
                if email:
                    error_result["email"] = email
                    
                return jsonify(error_result), 500
            
            # Determine classifier type
            classifier_type = "claude-llm" if classification else None
            
            # Save to Snowflake and Vector DB (always save, even for reclassifications)
            save_to_snowflake(
                domain=domain, 
                url=url, 
                content=content, 
                classification=classification, 
                snowflake_conn=snowflake_conn,
                crawler_type=crawler_type,
                classifier_type=classifier_type
            )
            
            # Process the fresh classification result
            result = process_fresh_result(classification, domain, email, url)

            # Add crawler_type to the result if not already included
            if crawler_type and "crawler_type" not in result:
                result["crawler_type"] = crawler_type
                
            # Add classifier_type to the result if not already included
            if classifier_type and "classifier_type" not in result:
                result["classifier_type"] = classifier_type

            # Ensure result consistency
            result = validate_result_consistency(result, domain)
            
            # Add company description if not already present
            if "company_description" not in result:
                result["company_description"] = extract_company_description(content, result.get("explanation", ""), domain)
            
            # Log the response for debugging
            logger.info(f"Sending fresh response to client")
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
            # Try to identify the error type if possible
            error_type, error_detail = detect_error_type(str(e))
            error_result = create_error_result(
                domain if 'domain' in locals() else "unknown", 
                error_type, 
                error_detail, 
                email if 'email' in locals() else None,
                crawler_type if 'crawler_type' in locals() else "exception_handler"
            )
            error_result["error"] = str(e)  # Add the actual error message
            if 'url' in locals():
                error_result["website_url"] = url
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
            from flask import request as flask_request
            
            # Store the original json
            original_json = flask_request.json
            
            # Create a class to simulate the request object
            class RequestProxy:
                @property
                def json(self):
                    return new_data
                    
                @property
                def method(self):
                    return 'POST'
            
            # Replace the request object temporarily
            temp_request = flask_request
            request = RequestProxy()
            
            try:
                # Call classify_domain with our proxy request
                result = classify_domain()
                return result
            finally:
                # Restore the original request
                request = temp_request
                
        except Exception as e:
            logger.error(f"Error processing email classification request: {e}\n{traceback.format_exc()}")
            error_type, error_detail = detect_error_type(str(e))
            error_result = create_error_result("unknown", error_type, error_detail, None, "email_handler")
            error_result["error"] = str(e)
            return jsonify(error_result), 500
            
    return app
