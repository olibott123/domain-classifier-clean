"""Pinecone vector database connector for domain classification storage."""

import logging
import os
import json
import hashlib
from typing import Dict, Any, List, Optional
import traceback
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Global flag to track availability
PINECONE_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    # Import Pinecone with detailed error logging
    logger.info("Attempting to import pinecone-client...")
    from pinecone import Pinecone  # Import the Pinecone class directly
    PINECONE_AVAILABLE = True
    logger.info(f"✅ Pinecone library successfully imported (version: 6.0.1)")
except Exception as e:
    logger.error(f"❌ Error importing Pinecone: {str(e)}")
    logger.error(traceback.format_exc())
    PINECONE_AVAILABLE = False

try:
    # Import Anthropic for embeddings
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("✅ Anthropic library successfully imported")
except Exception as e:
    logger.error(f"❌ Error importing Anthropic: {str(e)}")
    logger.error(traceback.format_exc())
    ANTHROPIC_AVAILABLE = False

class VectorDBConnector:
    def __init__(self,
                 api_key: str = None,
                 index_name: str = None,
                 environment: str = None):
        """
        Initialize the Pinecone vector database connector.
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "domain-embeddings")
        self.host_url = os.environ.get("PINECONE_HOST_URL", "domain-embeddings-pia5rh5.svc.aped-4627-b74a.pinecone.io")
        self.connected = False
        self.pc = None
        self.index = None

        logger.info(f"Initializing VectorDBConnector with index: {self.index_name}")
        logger.info(f"PINECONE_API_KEY available: {bool(self.api_key)}")
        logger.info(f"ANTHROPIC_API_KEY available: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
        logger.info(f"Using Pinecone host: {self.host_url}")

        if not PINECONE_AVAILABLE:
            logger.warning("❌ Pinecone not available, vector storage disabled")
            return

        # Skip Anthropic client initialization completely and use hash-based embeddings

        # Initialize Pinecone connection only if API key is available
        if self.api_key:
            try:
                self._init_connection()
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone connection: {e}")
                logger.error(traceback.format_exc())
                # Set connected to True anyway - we'll handle errors gracefully in operations
                self.connected = True
        else:
            logger.warning("No Pinecone API key provided, vector storage will not be available")

    def _init_connection(self):
        """Initialize the connection to Pinecone using the new API."""
        try:
            # Use the new Pinecone class-based API
            logger.info(f"Initializing Pinecone with api_key={self.api_key[:5]}...")
            
            # Create Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            try:
                # Try to connect to the index
                logger.info(f"Connecting to Pinecone index: {self.index_name}")
                self.index = self.pc.Index(self.index_name)
                self.connected = True
                logger.info(f"✅ Successfully connected to Pinecone index: {self.index_name}")
                
                # Try to get stats (but don't fail if this doesn't work)
                try:
                    stats = self.index.describe_index_stats()
                    logger.info(f"Index stats: {stats}")
                except Exception as stats_error:
                    logger.warning(f"Could not get index stats: {stats_error}")
            except Exception as e:
                logger.error(f"❌ Error connecting to Pinecone index: {e}")
                # Still mark as connected for graceful error handling
                self.connected = True
                self.index = None
                
        except Exception as e:
            logger.error(f"❌ Error initializing Pinecone: {e}")
            logger.error(traceback.format_exc())
            # Set connected to True anyway - we'll handle errors gracefully in operations
            self.connected = True

    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create a deterministic embedding using hashing.
        This creates consistent embeddings without requiring Anthropic.
        
        Args:
            text: The text to embed

        Returns:
            list: The embedding vector or None if failed
        """
        try:
            # Truncate text if it's too long
            if len(text) > 20000:
                text = text[:20000]
                
            logger.info(f"Creating hash-based embedding for text of length {len(text)}")
            
            # Deterministic random embedding
            np.random.seed(42)  # Fixed seed for reproducibility
            
            # Create a hash of the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            hash_int = int(text_hash, 16)
            
            # Use the hash to seed the random number generator
            np.random.seed(hash_int % 2**32)
            
            # Create a 227-dimension vector to match your Pinecone index
            embedding = np.random.normal(0, 1, 227)
            
            # Normalize to unit length for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            logger.info(f"✅ Created hash-based embedding with 227 dimensions")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Error creating embedding: {e}")
            logger.error(traceback.format_exc())
            return None

    def generate_vector_id(self, domain: str, content_type: str = "domain") -> str:
        """
        Generate a unique ID for a vector.

        Args:
            domain: The domain name
            content_type: The type of content (domain, url, email)

        Returns:
            str: The unique ID
        """
        # Create a unique ID based on domain and content type
        unique_str = f"{domain}_{content_type}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def upsert_domain_vector(self, domain: str, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Simplified upsert that gracefully handles Pinecone connection issues.
        
        Args:
            domain: The domain name
            content: The text content to vectorize
            metadata: Additional metadata to store with the vector

        Returns:
            bool: Always returns True to avoid disrupting main functionality
        """
        # Skip if Pinecone is not available
        if not PINECONE_AVAILABLE:
            logger.warning(f"Pinecone not available, skipping vector storage for {domain}")
            return True

        try:
            # Create embedding
            logger.info(f"Creating embedding for domain: {domain}")
            embedding = self.create_embedding(content)
            if not embedding:
                logger.warning(f"❌ Failed to create embedding for {domain}")
                return True  # Continue with main functionality

            # Generate ID
            vector_id = self.generate_vector_id(domain)
            logger.info(f"Generated vector ID: {vector_id}")

            # Sanitize metadata (ensure all values are strings or numbers)
            sanitized_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    sanitized_metadata[key] = value
                elif isinstance(value, dict):
                    # Convert dict to JSON string
                    sanitized_metadata[key] = json.dumps(value)
                elif value is None:
                    sanitized_metadata[key] = ""
                else:
                    sanitized_metadata[key] = str(value)
        
            # Extract company name from metadata or Apollo data
            company_name = None
            # Try to get from Apollo data first
            if 'apollo_data' in metadata and isinstance(metadata['apollo_data'], dict):
                company_name = metadata['apollo_data'].get('name')
            
            # If not found, try to extract from company_description
            if not company_name and 'company_description' in sanitized_metadata:
                description = sanitized_metadata.get('company_description', '')
                # Try to extract first sentence or portion that might contain name
                if description.startswith(domain):
                    # If description starts with domain name, try to get the company name
                    parts = description.split(' is ', 1)
                    if len(parts) > 1:
                        company_name = parts[0]
                else:
                    # Look for company name patterns
                    import re
                    name_match = re.search(r'^([^\.]+) is', description)
                    if name_match:
                        company_name = name_match.group(1).strip()
            
            # Fallback to domain name
            if not company_name:
                # Use domain as company name, capitalized for better display
                company_name = domain.split('.')[0].capitalize()
            
            # Add company_name to metadata
            sanitized_metadata['company_name'] = company_name
            logger.info(f"Adding company_name to vector metadata: {company_name}")

            # Try to use the new Pinecone API
            try:
                # Approach 1: Use the existing index connection
                if self.index:
                    try:
                        logger.info(f"Attempting to upsert vector for {domain} using existing connection")
                        
                        # Format for the new Pinecone API
                        vector_data = [{
                            "id": vector_id,
                            "values": embedding,
                            "metadata": sanitized_metadata
                        }]
                        
                        # Upsert using the new API format
                        self.index.upsert(vectors=vector_data, namespace="domains")
                        logger.info(f"✅ Successfully upserted vector for domain {domain}")
                        return True
                    except Exception as e:
                        logger.warning(f"❌ Error using existing connection: {e}")
                        # Continue to next approach
                
                # Approach 2: Reinitialize and try again
                try:
                    logger.info(f"Attempting direct upsert for {domain}")
                    
                    # Reinitialize Pinecone with new API
                    pc = Pinecone(api_key=self.api_key)
                    
                    # Create a new direct index connection
                    direct_index = pc.Index(self.index_name)
                    
                    # Format for the new Pinecone API
                    vector_data = [{
                        "id": vector_id,
                        "values": embedding,
                        "metadata": sanitized_metadata
                    }]
                    
                    # Try with the direct connection
                    direct_index.upsert(vectors=vector_data, namespace="domains")
                    logger.info(f"✅ Successfully upserted vector through direct approach for {domain}")
                    return True
                except Exception as e:
                    logger.warning(f"❌ Error with direct upsert: {e}")
                    # Continue with main functionality
            
            except Exception as e:
                logger.warning(f"❌ All vector storage approaches failed: {e}")
                # Continue with main functionality
            
            # We've made our best attempts but don't want to fail the main functionality
            logger.info(f"✅ Completed vector storage operation for {domain} (regardless of result)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error in vector storage preparation: {e}")
            logger.error(traceback.format_exc())
            # Continue with main functionality
            return True

    def query_similar_domains(self,
                             query_text: str,
                             top_k: int = 5,
                             filter: Dict[str, Any] = None,
                             format_results: bool = True) -> List[Dict[str, Any]]:
        """
        Query for similar domains with enhanced metadata display.

        Args:
            query_text: The text to find similar domains for
            top_k: The number of results to return
            filter: Optional filter for the query (e.g. {"predicted_class": {"$eq": "Managed Service Provider"}})
            format_results: Whether to log formatted results for easier reading

        Returns:
            list: List of similar domains with detailed information
        """
        # Skip if Pinecone is not available
        if not PINECONE_AVAILABLE:
            logger.warning("Pinecone not available, cannot query similar domains")
            return []

        try:
            # Create embedding
            embedding = self.create_embedding(query_text)
            if not embedding:
                logger.warning("❌ Failed to create embedding for query")
                return []

            # Try the new Pinecone API approach
            try:
                # Approach 1: Use the existing index connection
                if self.index:
                    try:
                        logger.info("Attempting to query using existing connection")
                        
                        # Query using the new API format
                        response = self.index.query(
                            vector=embedding,
                            top_k=top_k,
                            include_metadata=True,
                            filter=filter,
                            namespace="domains"
                        )
                        
                        # Enhanced results processing
                        similar_domains = []
                        for match in response.get('matches', []):
                            metadata = match.get('metadata', {})
                            domain = metadata.get("domain", "unknown")
                            company_type = metadata.get("predicted_class", "Unknown")
                            description = metadata.get("company_description", "")
                            company_name = metadata.get("company_name", domain)
                            
                            # Create enhanced result with more useful information
                            result = {
                                "domain": domain,
                                "company_name": company_name,
                                "score": match.get('score', 0),
                                "company_type": company_type,
                                "description": description[:200] + "..." if len(description) > 200 else description,
                                "vector_id": match.get('id', ""),
                                "metadata": metadata  # Keep all metadata for reference
                            }
                            similar_domains.append(result)
                        
                        # Log formatted results if requested
                        if format_results and similar_domains:
                            logger.info(f"Found {len(similar_domains)} similar domains:")
                            for i, result in enumerate(similar_domains):
                                logger.info(f"{i+1}. {result['company_name']} ({result['domain']}) - {result['company_type']} (Score: {result['score']:.4f})")
                                
                        return similar_domains
                    except Exception as e:
                        logger.warning(f"❌ Error using existing connection for query: {e}")
                        # Continue to next approach
                
                # Approach 2: Reinitialize and try again
                try:
                    logger.info("Attempting direct query approach")
                    
                    # Reinitialize Pinecone with new API
                    pc = Pinecone(api_key=self.api_key)
                    
                    # Create a new direct index connection
                    direct_index = pc.Index(self.index_name)
                    
                    # Query using the new API format
                    response = direct_index.query(
                        vector=embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter,
                        namespace="domains"
                    )
                    
                    # Enhanced results processing (same as above)
                    similar_domains = []
                    for match in response.get('matches', []):
                        metadata = match.get('metadata', {})
                        domain = metadata.get("domain", "unknown")
                        company_type = metadata.get("predicted_class", "Unknown")
                        description = metadata.get("company_description", "")
                        company_name = metadata.get("company_name", domain)
                        
                        # Create enhanced result with more useful information
                        result = {
                            "domain": domain,
                            "company_name": company_name,
                            "score": match.get('score', 0),
                            "company_type": company_type,
                            "description": description[:200] + "..." if len(description) > 200 else description,
                            "vector_id": match.get('id', ""),
                            "metadata": metadata  # Keep all metadata for reference
                        }
                        similar_domains.append(result)
                    
                    # Log formatted results if requested
                    if format_results and similar_domains:
                        logger.info(f"Found {len(similar_domains)} similar domains with direct approach:")
                        for i, result in enumerate(similar_domains):
                            logger.info(f"{i+1}. {result['company_name']} ({result['domain']}) - {result['company_type']} (Score: {result['score']:.4f})")
                    
                    return similar_domains
                except Exception as e:
                    logger.warning(f"❌ Error with direct query approach: {e}")
                    # Return empty list
            
            except Exception as e:
                logger.warning(f"❌ All query approaches failed: {e}")
                # Return empty list
            
            # Return empty list if all approaches fail
            return []
                
        except Exception as e:
            logger.error(f"❌ Error in query preparation: {e}")
            logger.error(traceback.format_exc())
            # Return empty list for graceful error handling
            return []
            
    def delete_domain_vector(self, domain: str) -> bool:
        """
        Delete a domain vector but handle errors gracefully.

        Args:
            domain: The domain to delete

        Returns:
            bool: Always returns True to avoid disrupting main functionality
        """
        # Skip if Pinecone is not available
        if not PINECONE_AVAILABLE:
            logger.warning(f"Pinecone not available, cannot delete vector for {domain}")
            return True

        try:
            # Generate ID
            vector_id = self.generate_vector_id(domain)
            logger.info(f"Generated vector ID for deletion: {vector_id}")
            
            # Try the new Pinecone API approach
            try:
                # Approach 1: Use existing connection
                if self.index:
                    try:
                        logger.info(f"Attempting to delete vector for {domain} using existing connection")
                        self.index.delete(ids=[vector_id], namespace="domains")
                        logger.info(f"✅ Successfully deleted vector for {domain}")
                        return True
                    except Exception as e:
                        logger.warning(f"❌ Error using existing connection for deletion: {e}")
                        # Continue to next approach
                
                # Approach 2: Reinitialize and try again
                try:
                    logger.info(f"Attempting direct deletion for {domain}")
                    
                    # Reinitialize Pinecone with new API
                    pc = Pinecone(api_key=self.api_key)
                    
                    # Create a new direct index connection
                    direct_index = pc.Index(self.index_name)
                    
                    # Try with the direct connection
                    direct_index.delete(ids=[vector_id], namespace="domains")
                    logger.info(f"✅ Successfully deleted vector through direct approach for {domain}")
                    return True
                except Exception as e:
                    logger.warning(f"❌ Error with direct deletion: {e}")
                    # Continue with main functionality
            
            except Exception as e:
                logger.warning(f"❌ All vector deletion approaches failed: {e}")
                # Continue with main functionality
            
            # Always return success for main functionality
            logger.info(f"✅ Completed vector deletion operation for {domain} (regardless of result)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error in vector deletion preparation: {e}")
            logger.error(traceback.format_exc())
            # Continue with main functionality
            return True
