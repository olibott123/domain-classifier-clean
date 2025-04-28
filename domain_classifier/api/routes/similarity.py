import logging
import traceback
from flask import request, jsonify

# Import services
from domain_classifier.storage.operations import query_similar_domains

# Set up logging
logger = logging.getLogger(__name__)

def register_similarity_routes(app, snowflake_conn):
    """Register similarity search related routes."""
    
    @app.route('/query-similar-domains', methods=['POST', 'OPTIONS'])
    def find_similar_domains():
        """Find domains similar to the given query text or domain"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            
            # Get query parameters
            query_text = data.get('query_text', '').strip()
            domain = data.get('domain', '').strip()
            top_k = data.get('top_k', 5)
            filter_criteria = data.get('filter', None)
            
            # Validate input
            if not query_text and not domain:
                return jsonify({"error": "Either query_text or domain must be provided"}), 400
                
            # If domain is provided but not query_text, get domain content for the query
            if domain and not query_text:
                # Get domain content from Snowflake
                try:
                    content = snowflake_conn.get_domain_content(domain)
                    if content:
                        query_text = content
                    else:
                        return jsonify({
                            "error": "No content found for domain",
                            "domain": domain
                        }), 404
                except Exception as e:
                    logger.error(f"Error getting domain content for similarity query: {e}")
                    return jsonify({
                        "error": f"Failed to retrieve content for domain: {str(e)}",
                        "domain": domain
                    }), 500
            
            # Query for similar domains
            results = query_similar_domains(
                query_text=query_text,
                top_k=top_k,
                filter=filter_criteria
            )
            
            # Return the results
            return jsonify({
                "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                "domain": domain if domain else None,
                "top_k": top_k,
                "results": results,
                "result_count": len(results)
            }), 200
            
        except Exception as e:
            logger.error(f"Error querying similar domains: {e}\n{traceback.format_exc()}")
            return jsonify({
                "error": str(e),
                "results": []
            }), 500
            
    return app
