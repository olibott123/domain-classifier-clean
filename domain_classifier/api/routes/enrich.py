"""Enrichment related routes for the domain classifier API."""
import logging
import traceback
from flask import request, jsonify, current_app

# Import utilities
from domain_classifier.utils.error_handling import detect_error_type, create_error_result, is_domain_worth_crawling
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.utils.final_classification import determine_final_classification
from domain_classifier.classifiers.decision_tree import create_parked_domain_result, is_parked_domain
from domain_classifier.storage.result_processor import process_fresh_result
from domain_classifier.enrichment.ai_data_extractor import extract_company_data_from_content

# Set up logging
logger = logging.getLogger(__name__)

def register_enrich_routes(app, snowflake_conn):
    """Register enrichment related routes."""
    
    @app.route('/classify-and-enrich', methods=['POST', 'OPTIONS'])
    def classify_and_enrich():
        """Classify a domain and enrich it with Apollo data"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            input_value = data.get('url', '').strip()
            
            if not input_value:
                return jsonify({"error": "URL or email is required"}), 400
            
            # Determine if input is an email
            is_email = '@' in input_value
            email = input_value if is_email else None
            
            # Extract domain for checking
            domain = None
            if is_email and '@' in input_value:
                domain = input_value.split('@')[-1].strip().lower()
            else:
                # Basic domain extraction for URL
                domain = input_value.replace('https://', '').replace('http://', '')
                if '/' in domain:
                    domain = domain.split('/', 1)[0]
                if domain.startswith('www.'):
                    domain = domain[4:]
                    
            # Create URL for checks and displaying
            url = f"https://{domain}"
                
            # Direct check if domain is worth crawling or is parked
            worth_crawling, has_dns, dns_error, potentially_flaky = is_domain_worth_crawling(domain)
            
            # Check for DNS resolution failure
            if not has_dns and dns_error != "parked_domain":
                logger.info(f"Domain {domain} has DNS resolution issues: {dns_error}")
                
                # Create an error result
                error_result = create_error_result(
                    domain,
                    "dns_error",
                    dns_error,
                    email,
                    "early_check"
                )
                error_result["website_url"] = url
                error_result["final_classification"] = "7-No Website available"
                return jsonify(error_result), 200  # Return 200 instead of 503
                
            # Check for parked domain
            if dns_error == "parked_domain" or not worth_crawling:
                logger.info(f"Domain {domain} detected as parked domain during initial check")
                
                # Create a proper parked domain result
                parked_result = create_parked_domain_result(domain, crawler_type="early_check_parked")
                
                # Process it through the normal result processor
                result = process_fresh_result(parked_result, domain, email, url)
                
                # Ensure proper classification and fields
                result["final_classification"] = "6-Parked Domain - no enrichment"
                result["crawler_type"] = "early_check_parked"
                result["classifier_type"] = "early_detection"
                result["is_parked"] = True
                
                # Add email and URL if provided
                if email:
                    result["email"] = email
                result["website_url"] = url
                
                return jsonify(result), 200  # Return 200 instead of 503
            
            # Check for early parked domain detection using direct crawl
            try:
                from domain_classifier.crawlers.direct_crawler import direct_crawl
                logger.info(f"Performing quick check for parked domain before enriching: {domain}")
                quick_check_content, (error_type, error_detail), quick_crawler_type = direct_crawl(url, timeout=5.0)
                
                # Check if this is a parked domain
                if error_type == "is_parked" or (quick_check_content and is_parked_domain(quick_check_content, domain)):
                    logger.info(f"Quick check detected parked domain: {domain}")
                    parked_result = create_parked_domain_result(domain, crawler_type="quick_check_parked")
                    
                    # Process the result through the normal result processor
                    result = process_fresh_result(parked_result, domain, email, url)
                    
                    # Ensure proper classification and fields
                    result["final_classification"] = "6-Parked Domain - no enrichment"
                    result["crawler_type"] = "quick_check_parked"
                    result["classifier_type"] = "early_detection"
                    result["is_parked"] = True
                    
                    # Add email and URL if provided
                    if email:
                        result["email"] = email
                    result["website_url"] = url
                    
                    return jsonify(result), 200
            except Exception as e:
                logger.warning(f"Early parked domain check failed in enrichment route: {e}")
            
            # First perform standard classification by making an internal request
            # We'll use the routes directly from the app, rather than importing functions
            
            # Get the classify_domain function from app's view functions
            from domain_classifier.api.routes.classify import register_classify_routes
            
            # Call the registered classify-domain route directly using the Flask test client
            with app.test_client() as client:
                response = client.post('/classify-domain', json=data)
                classification_result = response.get_json()
                status_code = response.status_code
            
            # Only proceed with enrichment for successful classifications
            if status_code >= 400:
                logger.warning(f"Classification failed with status {status_code}, skipping enrichment")
                
                # Ensure final_classification is set even for error results
                if "final_classification" not in classification_result:
                    classification_result["final_classification"] = determine_final_classification(classification_result)
                    
                return jsonify(classification_result), 200  # Return 200 instead of status_code
            
            # Extract domain, email and crawler type from classification result
            domain = classification_result.get('domain')
            email = classification_result.get('email')  # This will be set if the input was an email
            crawler_type = classification_result.get('crawler_type')  # Get crawler type from classification result
            
            if not domain:
                logger.error("No domain found in classification result")
                return jsonify({"error": "Failed to extract domain from classification result"}), 200  # Return 200 instead of 500
            
            # Import Apollo connector here to avoid circular imports
            from domain_classifier.enrichment.apollo_connector import ApolloConnector
            from domain_classifier.enrichment.description_enhancer import enhance_company_description, generate_detailed_description
            
            # Initialize Apollo connector
            apollo = ApolloConnector()
            
            # Enrich with Apollo company data
            company_data = apollo.enrich_company(domain)
            
            # Don't look up person data to save Apollo credits
            person_data = None
            
            # Get the website content for AI extraction
            website_content = snowflake_conn.get_domain_content(domain)
            
            # Always attempt AI extraction, regardless of Apollo data
            logger.info(f"Attempting AI extraction for {domain}")
            
            # Extract company data using AI from the website content
            if website_content:
                ai_company_data = extract_company_data_from_content(
                    website_content, 
                    domain, 
                    classification_result
                )
                
                # Add the AI-extracted data to the result
                if ai_company_data:
                    logger.info(f"Successfully extracted AI company data for {domain}")
                    classification_result["ai_company_data"] = ai_company_data
            else:
                logger.warning(f"No website content available for AI extraction for {domain}")
            
            # Import recommendation engine
            from domain_classifier.enrichment.recommendation_engine import DomotzRecommendationEngine
            
            # Generate recommendations based on classification and Apollo data
            recommendation_engine = DomotzRecommendationEngine()
            recommendations = recommendation_engine.generate_recommendations(
                classification_result.get('predicted_class'),
                company_data
            )
            
            # Step 1: First enhance with Apollo data
            if company_data:
                logger.info(f"Enhancing description with Apollo data for {domain}")
                basic_enhanced_description = enhance_company_description(
                    classification_result.get("company_description", ""),
                    company_data,
                    classification_result
                )
                classification_result["company_description"] = basic_enhanced_description
            
            # Step 2: Then use Claude to generate a more detailed description
            try:
                detailed_description = generate_detailed_description(
                    classification_result,
                    company_data,
                    None  # No person data passed
                )
                
                if detailed_description and len(detailed_description) > 50:
                    classification_result["company_description"] = detailed_description
                    logger.info(f"Updated description with detailed Claude-generated version for {domain}")
            except Exception as desc_error:
                logger.error(f"Error generating detailed description: {desc_error}")
                # Keep the basic enhanced description if the detailed one fails
            
            # Add enrichment data to classification result
            classification_result['apollo_data'] = company_data or {}
            
            # Add recommendations
            classification_result['domotz_recommendations'] = recommendations
            
            # Make sure the crawler_type is preserved from the original classification
            if not classification_result.get('crawler_type') and crawler_type:
                classification_result['crawler_type'] = crawler_type
                
            # Update final_classification based on Apollo data
            # For parked domains, check if we need to update from 6-Parked Domain - no enrichment to 5-Parked Domain with partial enrichment
            if classification_result.get('final_classification') == "6-Parked Domain - no enrichment" and company_data:
                classification_result['final_classification'] = "5-Parked Domain with partial enrichment"
                logger.info(f"Updated final classification to 5-Parked Domain with partial enrichment for {domain}")
            elif "final_classification" not in classification_result:
                # Ensure final_classification is set
                classification_result["final_classification"] = determine_final_classification(classification_result)
                logger.info(f"Added final classification: {classification_result['final_classification']} for {domain}")
                
            # Save the enriched classification to Snowflake
            url = f"https://{domain}"
            content = snowflake_conn.get_domain_content(domain)
            
            # Save the enhanced data to Snowflake (with Apollo data)
            save_to_snowflake(
                domain=domain, 
                url=url, 
                content=content, 
                classification=classification_result,
                snowflake_conn=snowflake_conn,
                apollo_company_data=company_data,
                crawler_type=crawler_type,  # Explicitly pass the crawler_type from the original classification
                classifier_type="claude-llm-enriched"
            )
            
            # Return the enriched result
            logger.info(f"Successfully enriched and generated recommendations for {domain}")
            return jsonify(classification_result), 200
            
        except Exception as e:
            logger.error(f"Error in classify-and-enrich: {e}\n{traceback.format_exc()}")
            # Try to identify the error type
            error_type, error_detail = detect_error_type(str(e))
            # Create an error response
            error_result = create_error_result(
                domain if 'domain' in locals() else "unknown", 
                error_type, 
                error_detail,
                email if 'email' in locals() else None,
                "enrich_error_handler"  # Set a crawler_type for enrichment errors
            )
            error_result["error"] = str(e)  # Add the actual error message
            
            # Ensure final_classification is set for error results
            if "final_classification" not in error_result:
                error_result["final_classification"] = determine_final_classification(error_result)
                
            return jsonify(error_result), 200  # Return 200 instead of 500
            
    return app

def _is_minimal_apollo_data(apollo_data):
    """Check if Apollo data is minimal and needs enhancement."""
    # Define the essential fields we want to check
    essential_fields = ["name", "address", "industry", "employee_count", "phone"]
    
    # Count how many essential fields are missing
    missing_fields = sum(1 for field in essential_fields if not apollo_data.get(field))
    
    # If most essential fields are missing, consider it minimal
    return missing_fields >= 3
