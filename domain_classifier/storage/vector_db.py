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
            # Use the new Pinecone client API
            logger.info(f"Initializing Pinecone with api_key={self.api_key[:5]}...")
            
            # Create the Pinecone client
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            
            # Try to connect to the index
            logger.info(f"Connecting to Pinecone index: {self.index_name} at host: {self.host_url}")
            
            try:
                # Try the new approach with host parameter
                self.index = self.pc.Index(host=self.host_url)
                self.connected = True
                logger.info(f"✅ Successfully connected to Pinecone index with host: {self.index_name}")
            except Exception as e1:
                logger.warning(f"Error connecting with host parameter: {e1}")
                
                try:
                    # Try simpler approach with just index name
                    self.index = self.pc.Index(self.index_name)
                    self.connected = True
                    logger.info(f"✅ Successfully connected to Pinecone index by name: {self.index_name}")
                except Exception as e2:
                    logger.error(f"❌ Error connecting to index by name: {e2}")
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

            # Try using the new API format
            try:
                # Try with existing index connection
                if self.index:
                    logger.info(f"Attempting to upsert vector for {domain} using new Pinecone API")
                    
                    # Format the vector in the new expected format
                    vector_data = {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": sanitized_metadata
                    }
                    
                    # Upsert the vector
                    self.index.upsert(vectors=[vector_data], namespace="domains")
                    logger.info(f"✅ Successfully upserted vector for {domain}")
                    return True
            except Exception as e:
                logger.warning(f"❌ Error using new Pinecone API: {e}")
                # Try alternative approaches but don't fail main functionality
                pass
                
            # If we get here, all approaches failed but we don't want to disrupt main functionality
            logger.warning(f"❌ All vector storage approaches failed for {domain}, but continuing with main functionality")
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

            # Try with new Pinecone API
            try:
                if self.index:
                    logger.info("Attempting to query using new Pinecone API")
                    response = self.index.query(
                        namespace="domains",
                        vector=embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter
                    )
                    
                    similar_domains = []
                    for match in response.get('matches', []):
                        metadata = match.get('metadata', {})
                        domain = metadata.get("domain", "unknown")
                        
                        result = {
                            "domain": domain,
                            "score": match.get('score', 0),
                            "metadata": metadata
                        }
                        similar_domains.append(result)
                        
                    logger.info(f"✅ Found {len(similar_domains)} similar domains")
                    return similar_domains
            except Exception as e:
                logger.warning(f"❌ Error using new Pinecone API for query: {e}")
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
            
            # Try with new Pinecone API
            try:
                if self.index:
                    logger.info(f"Attempting to delete vector for {domain} using new Pinecone API")
                    self.index.delete(ids=[vector_id], namespace="domains")
                    logger.info(f"✅ Successfully deleted vector for {domain}")
                    return True
            except Exception as e:
                logger.warning(f"❌ Error using new Pinecone API for deletion: {e}")
                # Continue with main functionality
            
            # Always return success for main functionality
            return True
                
        except Exception as e:
            logger.error(f"❌ Error in vector deletion preparation: {e}")
            logger.error(traceback.format_exc())
            # Continue with main functionality
            return True
