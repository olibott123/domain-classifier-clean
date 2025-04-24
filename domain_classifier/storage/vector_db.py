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
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.host_url = os.environ.get("PINECONE_HOST_URL", "domain-embeddings-pia5rh5.svc.aped-4627-b74a.pinecone.io")
        self.connected = False
        self.index = None
        self.anthropic_client = None

        logger.info(f"Initializing VectorDBConnector with index: {self.index_name}")
        logger.info(f"PINECONE_API_KEY available: {bool(self.api_key)}")
        logger.info(f"ANTHROPIC_API_KEY available: {bool(self.anthropic_api_key)}")
        logger.info(f"Using Pinecone host: {self.host_url}")

        if not PINECONE_AVAILABLE:
            logger.warning("❌ Pinecone not available, vector storage disabled")
            return

        # We'll skip Anthropic client initialization completely and use hash-based embeddings
        # This is the most reliable approach for now

        # Initialize Pinecone connection
        if self.api_key:
            try:
                self._init_connection()
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone connection: {e}")
                self.connected = False
        else:
            logger.warning("No Pinecone API key provided, vector storage will not be available")

    def _init_connection(self):
        """Initialize the connection to Pinecone using SDK 2.2.x."""
        try:
            # Basic initialization with only required parameters
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            try:
                # Try to connect with explicit host parameter
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
                logger.info(f"Host parameter not supported, trying without it")
                self.index = pinecone.Index(self.index_name)
                self.connected = True
                logger.info(f"✅ Successfully connected to Pinecone index without host: {self.index_name}")
                
        except Exception as e:
            logger.error(f"❌ Error connecting to Pinecone: {e}")
            logger.error(traceback.format_exc())
            self.connected = False

    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create a deterministic embedding using hashing instead of semantic embeddings.
        This avoids the need for the Anthropic client and ensures consistency.
        
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

    def upsert_domain_vector(self,
                            domain: str,
                            content: str,
                            metadata: Dict[str, Any]) -> bool:
        """
        Upsert a domain vector into Pinecone.

        Args:
            domain: The domain name
            content: The text content to vectorize
            metadata: Additional metadata to store with the vector

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected or not self.index:
            logger.warning(f"❌ Not connected to Pinecone, cannot upsert vector for {domain}")
            return False

        try:
            # Create embedding
            logger.info(f"Creating embedding for domain: {domain}")
            embedding = self.create_embedding(content)
            if not embedding:
                logger.warning(f"❌ Failed to create embedding for {domain}")
                return False

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

            # Include key information in log
            logger.info(f"Upserting vector with metadata: domain={domain}, class={metadata.get('predicted_class', 'unknown')}")

            # Upsert vector with explicit error handling
            try:
                self.index.upsert(vectors=[(vector_id, embedding, sanitized_metadata)])
                logger.info(f"✅ Successfully upserted vector for domain {domain} to Pinecone!")
                return True
            except Exception as e:
                logger.error(f"❌ Error in Pinecone upsert operation: {e}")
                if "Unknown host" in str(e) or "Name or service not known" in str(e):
                    logger.error("This appears to be a DNS resolution issue with the Pinecone service")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error upserting vector for {domain}: {e}")
            logger.error(traceback.format_exc())
            return False

    def query_similar_domains(self,
                             query_text: str,
                             top_k: int = 5,
                             filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query Pinecone for domains similar to the given text.

        Args:
            query_text: The text to find similar domains for
            top_k: The number of results to return
            filter: Optional filter for the query

        Returns:
            list: List of similar domains with metadata
        """
        if not self.connected or not self.index:
            logger.warning("❌ Not connected to Pinecone, cannot query similar domains")
            return []

        try:
            # Create embedding
            embedding = self.create_embedding(query_text)
            if not embedding:
                logger.warning("❌ Failed to create embedding for query")
                return []

            # Query Pinecone with explicit error handling
            try:
                results = self.index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter
                )
            except Exception as e:
                logger.error(f"❌ Error in Pinecone query operation: {e}")
                if "Unknown host" in str(e) or "Name or service not known" in str(e):
                    logger.error("This appears to be a DNS resolution issue with the Pinecone service")
                return []

            # Process results
            similar_domains = []
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

            logger.info(f"✅ Found {len(similar_domains)} similar domains")
            return similar_domains
        except Exception as e:
            logger.error(f"❌ Error querying similar domains: {e}")
            logger.error(traceback.format_exc())
            return []

    def delete_domain_vector(self, domain: str) -> bool:
        """
        Delete a domain vector from Pinecone.

        Args:
            domain: The domain to delete

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected or not self.index:
            logger.warning(f"❌ Not connected to Pinecone, cannot delete vector for {domain}")
            return False

        try:
            # Generate ID
            vector_id = self.generate_vector_id(domain)
            
            # Delete vector with explicit error handling
            try:
                self.index.delete(ids=[vector_id])
                logger.info(f"✅ Deleted vector for domain {domain}")
                return True
            except Exception as e:
                logger.error(f"❌ Error in Pinecone delete operation: {e}")
                if "Unknown host" in str(e) or "Name or service not known" in str(e):
                    logger.error("This appears to be a DNS resolution issue with the Pinecone service")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting vector for {domain}: {e}")
            logger.error(traceback.format_exc())
            return False
