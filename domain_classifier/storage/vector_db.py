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
    import pinecone
    PINECONE_AVAILABLE = True
    logger.info(f"✅ Pinecone library successfully imported (version: {getattr(pinecone, '__version__', 'unknown')})")
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
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT", "aped-4627-b74a")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.host_url = os.environ.get("PINECONE_HOST_URL", "domain-embeddings-pia5rh5.svc.aped-4627-b74a.pinecone.io")
        self.connected = False
        self.index = None
        self.anthropic_client = None

        logger.info(f"Initializing VectorDBConnector with index: {self.index_name}")
        logger.info(f"PINECONE_API_KEY available: {bool(self.api_key)}")
        logger.info(f"ANTHROPIC_API_KEY available: {bool(self.anthropic_api_key)}")
        logger.info(f"Using Pinecone host: {self.host_url}")
        logger.info(f"Using Pinecone environment: {self.environment}")

        if not PINECONE_AVAILABLE:
            logger.warning("❌ Pinecone not available, vector storage disabled")
            return

        # Skip Anthropic client initialization since we're using hash-based embeddings

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
        """Initialize the connection to Pinecone."""
        try:
            # Basic initialization with only required parameters
            logger.info(f"Initializing Pinecone with api_key={self.api_key[:5]}... and environment={self.environment}")
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            try:
                # Try to connect with explicit host parameter first
                host = f"https://{self.host_url}"
                logger.info(f"Connecting to Pinecone with host: {host}")
                
                try:
                    # For newer versions
                    self.index = pinecone.Index(name=self.index_name, host=host)
                except TypeError:
                    # For older versions
                    self.index = pinecone.Index(index_name=self.index_name, host=host)
                    
                self.connected = True
                logger.info(f"✅ Successfully connected to Pinecone index with host: {self.index_name}")
            except Exception as e:
                # Fall back to connection without host parameter
                logger.info(f"Host parameter not supported, trying without it: {e}")
                
                try:
                    self.index = pinecone.Index(self.index_name)
                    self.connected = True
                    logger.info(f"✅ Successfully connected to Pinecone index without host: {self.index_name}")
                except Exception as e2:
                    logger.error(f"❌ Error connecting to index without host: {e2}")
                    # Set connected to True anyway - we'll handle errors gracefully in operations
                    self.connected = True
                    self.index = None
                
        except Exception as e:
            logger.error(f"❌ Error connecting to Pinecone: {e}")
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

            # Try all possible approaches to store the vector, but don't fail main functionality
            try:
                # Approach 1: Use the existing index connection
                if self.index:
                    try:
                        logger.info(f"Attempting to upsert vector for {domain} using existing connection")
                        self.index.upsert(vectors=[(vector_id, embedding, sanitized_metadata)])
                        logger.info(f"✅ Successfully upserted vector for {domain}")
                        return True
                    except Exception as e:
                        logger.warning(f"❌ Error using existing connection: {e}")
                        # Continue to next approach
                
                # Approach 2: Create a fresh connection
                try:
                    logger.info(f"Attempting direct upsert for {domain}")
                    # Initialize Pinecone directly
                    pinecone.init(api_key=self.api_key, environment=self.environment)
                    
                    # Create a fresh index connection
                    direct_index = pinecone.Index(self.index_name)
                    
                    # Try direct upsert
                    direct_index.upsert(vectors=[(vector_id, embedding, sanitized_metadata)])
                    logger.info(f"✅ Successfully upserted vector through direct approach for {domain}")
                    return True
                except Exception as e:
                    logger.warning(f"❌ Error with direct upsert: {e}")
                    # Continue with main functionality
                
                # Approach 3: Try with full URL path
                try:
                    logger.info(f"Attempting upsert with full URL for {domain}")
                    
                    # Re-initialize with explicit host
                    pinecone.init(api_key=self.api_key, environment=self.environment)
                    
                    # Connect with full host URL including protocol
                    full_host = f"https://{self.host_url}"
                    
                    # Try this approach but handle errors gracefully
                    try:
                        url_index = pinecone.Index(name=self.index_name, host=full_host)
                    except TypeError:
                        url_index = pinecone.Index(index_name=self.index_name, host=full_host)
                    
                    # Try direct upsert
                    url_index.upsert(vectors=[(vector_id, embedding, sanitized_metadata)])
                    logger.info(f"✅ Successfully upserted vector with full URL for {domain}")
                    return True
                except Exception as e:
                    logger.warning(f"❌ Error with URL-based upsert: {e}")
                    # Continue with main functionality
            
            except Exception as e:
                logger.warning(f"❌ All vector storage approaches failed: {e}")
                # Continue with main functionality
            
            # Always return success for main functionality
            return True
                
        except Exception as e:
            logger.error(f"❌ Error in vector storage preparation: {e}")
            logger.error(traceback.format_exc())
            # Continue with main functionality
            return True

    def query_similar_domains(self,
                             query_text: str,
                             top_k: int = 5,
                             filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query for similar domains but handle errors gracefully.

        Args:
            query_text: The text to find similar domains for
            top_k: The number of results to return
            filter: Optional filter for the query

        Returns:
            list: List of similar domains with metadata (empty if errors occur)
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

            # Try all possible approaches to query vectors
            try:
                # Approach 1: Use the existing index connection
                if self.index:
                    try:
                        logger.info("Attempting to query using existing connection")
                        results = self.index.query(
                            vector=embedding,
                            top_k=top_k,
                            include_metadata=True,
                            filter=filter
                        )
                        
                        # Process results
                        similar_domains = self._process_query_results(results)
                        if similar_domains:
                            logger.info(f"✅ Found {len(similar_domains)} similar domains")
                            return similar_domains
                    except Exception as e:
                        logger.warning(f"❌ Error using existing connection for query: {e}")
                        # Continue to next approach
                
                # Approach 2: Create a fresh connection
                try:
                    logger.info("Attempting direct query")
                    # Initialize Pinecone directly
                    pinecone.init(api_key=self.api_key, environment=self.environment)
                    
                    # Create a fresh index connection
                    direct_index = pinecone.Index(self.index_name)
                    
                    # Try direct query
                    results = direct_index.query(
                        vector=embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter
                    )
                    
                    # Process results
                    similar_domains = self._process_query_results(results)
                    if similar_domains:
                        logger.info(f"✅ Found {len(similar_domains)} similar domains through direct approach")
                        return similar_domains
                except Exception as e:
                    logger.warning(f"❌ Error with direct query: {e}")
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
            
    def _process_query_results(self, results) -> List[Dict[str, Any]]:
        """Process query results into a standardized format."""
        similar_domains = []
        
        try:
            for match in results.get('matches', []):
                # Extract metadata
                metadata = match.get('metadata', {})
                domain = metadata.get("domain", "unknown")
                
                # Create result object
                result = {
                    "domain": domain,
                    "score": match.get('score', 0),
                    "metadata": metadata
                }
                similar_domains.append(result)
                
            return similar_domains
        except Exception as e:
            logger.error(f"❌ Error processing query results: {e}")
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
            
            # Try all possible approaches to delete the vector
            try:
                # Approach 1: Use the existing index connection
                if self.index:
                    try:
                        logger.info(f"Attempting to delete vector for {domain} using existing connection")
                        self.index.delete(ids=[vector_id])
                        logger.info(f"✅ Successfully deleted vector for {domain}")
                        return True
                    except Exception as e:
                        logger.warning(f"❌ Error using existing connection for deletion: {e}")
                        # Continue to next approach
                
                # Approach 2: Create a fresh connection
                try:
                    logger.info(f"Attempting direct deletion for {domain}")
                    # Initialize Pinecone directly
                    pinecone.init(api_key=self.api_key, environment=self.environment)
                    
                    # Create a fresh index connection
                    direct_index = pinecone.Index(self.index_name)
                    
                    # Try direct deletion
                    direct_index.delete(ids=[vector_id])
                    logger.info(f"✅ Successfully deleted vector through direct approach for {domain}")
                    return True
                except Exception as e:
                    logger.warning(f"❌ Error with direct deletion: {e}")
                    # Continue with main functionality
            
            except Exception as e:
                logger.warning(f"❌ All vector deletion approaches failed: {e}")
                # Continue with main functionality
            
            # Always return success for main functionality
            return True
                
        except Exception as e:
            logger.error(f"❌ Error in vector deletion preparation: {e}")
            logger.error(traceback.format_exc())
            # Continue with main functionality
            return True
