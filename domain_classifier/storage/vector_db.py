"""Pinecone vector database connector for domain classification storage."""
import logging
import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import traceback
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Global flag to track availability
PINECONE_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    # Import Pinecone with detailed error logging (for older version)
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
        
        Args:
            api_key: The API key for Pinecone
            index_name: The name of the Pinecone index to use
            environment: The Pinecone environment (region) to use
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        # Use domain-embeddings to match existing index
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "domain-embeddings")
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.connected = False
        self.index = None
        self.anthropic_client = None
        
        logger.info(f"Initializing VectorDBConnector with index: {self.index_name}")
        logger.info(f"PINECONE_API_KEY available: {bool(self.api_key)}")
        logger.info(f"ANTHROPIC_API_KEY available: {bool(self.anthropic_api_key)}")
        logger.info(f"Using Pinecone environment: {self.environment}")
        
        # Don't even try if dependencies aren't available
        if not PINECONE_AVAILABLE or not ANTHROPIC_AVAILABLE:
            logger.warning("❌ Pinecone or Anthropic not available, vector storage disabled")
            return
            
        # Set up Anthropic client if API key is available
        if self.anthropic_api_key:
            try:
                # Fix Anthropic client initialization - try both modern and legacy formats
                logger.info("Initializing Anthropic client (with compatibility handling)...")
                try:
                    # Try modern client init with just the api_key parameter
                    self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                    logger.info("✅ Modern Anthropic client initialized successfully")
                except TypeError as e:
                    # If that fails with TypeError, try legacy format
                    logger.info(f"Modern client failed: {str(e)}, trying legacy format")
                    if hasattr(anthropic, "Client"):
                        self.anthropic_client = anthropic.Client(api_key=self.anthropic_api_key)
                        logger.info("✅ Legacy Anthropic client initialized successfully")
                    else:
                        raise ValueError("Neither modern nor legacy Anthropic client formats worked")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Anthropic client: {e}")
                logger.error(traceback.format_exc())
                self.anthropic_client = None
        else:
            logger.warning("No Anthropic API key provided, embeddings will not be available")
        
        # Initialize connection to Pinecone if API key is available
        if self.api_key:
            try:
                logger.info("Attempting to connect to Pinecone...")
                self._init_connection()
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone connection: {e}")
                logger.error(traceback.format_exc())
                self.connected = False
        else:
            logger.warning("No Pinecone API key provided, vector storage will not be available")
    
    def _init_connection(self):
        """Initialize the connection to Pinecone using older API."""
        try:
            # Initialize Pinecone client (older API style)
            logger.info(f"Initializing Pinecone connection with environment: {self.environment}")
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Check if index exists - safely
            try:
                logger.info("Listing available Pinecone indexes...")
                existing_indexes = pinecone.list_indexes()
                logger.info(f"Available Pinecone indexes: {existing_indexes}")
                
                # Log more details for debugging
                if not existing_indexes:
                    logger.warning("⚠️ No indexes found with the provided API key")
                    logger.info("This means either: 1) The API key doesn't have access to any indexes, or 2) You need to create an index")
            except Exception as e:
                logger.error(f"❌ Error listing Pinecone indexes: {e}")
                logger.error(traceback.format_exc())
                return
                
            # If index exists, connect to it
            if self.index_name in existing_indexes:
                logger.info(f"✅ Found existing Pinecone index: {self.index_name}")
                try:
                    # For the older API, we use this reference
                    self.index = pinecone.Index(self.index_name)
                    self.connected = True
                    logger.info("✅ Successfully connected to Pinecone index")
                    
                    # Try to get index stats to confirm connection is working
                    try:
                        stats = pinecone.describe_index(self.index_name)
                        dimension = stats.get("dimension", 0)
                        logger.info(f"Index dimension: {dimension}")
                        logger.info(f"Index stats: {stats}")
                    except Exception as e:
                        logger.warning(f"Could not get index stats: {e}")
                        
                except Exception as e:
                    logger.error(f"❌ Error connecting to index: {e}")
                    logger.error(traceback.format_exc())
                    return
            else:
                logger.warning(f"❌ Index {self.index_name} not found in available indexes: {existing_indexes}")
                logger.warning("Unable to connect to the index - please make sure it exists in your Pinecone account")
                logger.warning("You'll need to manually create the index in the Pinecone Console with:")
                logger.warning(f"- Name: {self.index_name}")
                logger.warning("- Dimensions: 227")
                logger.warning("- Metric: cosine")
                logger.warning(f"- Environment: {self.environment}")
                
                # Don't try to create it automatically as it failed before
                self.connected = False
                
        except Exception as e:
            logger.error(f"❌ Error connecting to Pinecone: {e}")
            logger.error(traceback.format_exc())
            self.connected = False
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create an embedding for the given text using a deterministic hash-based approach.
        
        Args:
            text: The text to embed
            
        Returns:
            list: The embedding vector or None if failed
        """
        try:
            # Truncate text if it's too long
            text_length = len(text)
            if text_length > 20000:
                logger.warning(f"Text too long ({text_length} chars), truncating to 20000 chars")
                text = text[:20000]
            
            logger.info(f"Creating embedding for text of length {len(text)} chars...")
            
            # Create a deterministic embedding based on the text content
            # This uses a hash-based approach that's consistent for the same input
            np.random.seed(42)  # Fixed seed for reproducibility
            
            # Create a hash of the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            hash_int = int(text_hash, 16)
            
            # Use the hash to seed the random number generator
            np.random.seed(hash_int % 2**32)
            
            # Create an embedding vector with 227 dimensions to match the index
            embedding = np.random.normal(0, 1, 227)
            
            # Normalize the embedding to unit length (required for cosine similarity)
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
            
            # Upsert vector (for older API)
            self.index.upsert(
                vectors=[(vector_id, embedding, sanitized_metadata)]
            )
            
            logger.info(f"✅ Successfully upserted vector for domain {domain} to Pinecone!")
            return True
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
            
            # Query Pinecone (for older API)
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            
            # Process results - older API has slightly different format
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
            
            # Delete vector
            self.index.delete(ids=[vector_id])
            
            logger.info(f"✅ Deleted vector for domain {domain}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting vector for {domain}: {e}")
            logger.error(traceback.format_exc())
            return False
