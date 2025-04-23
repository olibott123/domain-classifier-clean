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
    # Import Pinecone
    import pinecone
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
    logger.info("✅ Pinecone library successfully imported")
except ImportError:
    logger.warning("❌ Pinecone not available, vector storage will be disabled")
    
try:
    # Import Anthropic for embeddings
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("✅ Anthropic library successfully imported")
except ImportError:
    logger.warning("❌ Anthropic not available, embeddings will be disabled")

class VectorDBConnector:
    def __init__(self, 
                 api_key: str = None, 
                 index_name: str = None):
        """
        Initialize the Pinecone vector database connector.
        
        Args:
            api_key: The API key for Pinecone
            index_name: The name of the Pinecone index to use
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "domain-classification")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.connected = False
        self.client = None
        self.index = None
        self.anthropic_client = None
        
        logger.info(f"Initializing VectorDBConnector with index: {self.index_name}")
        logger.info(f"PINECONE_API_KEY available: {bool(self.api_key)}")
        logger.info(f"ANTHROPIC_API_KEY available: {bool(self.anthropic_api_key)}")
        
        # Don't even try if dependencies aren't available
        if not PINECONE_AVAILABLE or not ANTHROPIC_AVAILABLE:
            logger.warning("❌ Pinecone or Anthropic not available, vector storage disabled")
            return
            
        # Set up Anthropic client if API key is available
        if self.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.anthropic_api_key
                )
                logger.info("✅ Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Anthropic client: {e}")
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
                self.connected = False
        else:
            logger.warning("No Pinecone API key provided, vector storage will not be available")
    
    def _init_connection(self):
        """Initialize the connection to Pinecone."""
        try:
            # Initialize Pinecone client
            logger.info("Creating Pinecone client...")
            self.client = Pinecone(api_key=self.api_key)
            
            # Check if index exists - safely
            try:
                logger.info("Listing available Pinecone indexes...")
                existing_indexes = [index.name for index in self.client.list_indexes()]
                logger.info(f"Available Pinecone indexes: {existing_indexes}")
            except Exception as e:
                logger.error(f"❌ Error listing Pinecone indexes: {e}")
                return
            
            # If index exists, connect to it
            if self.index_name in existing_indexes:
                logger.info(f"✅ Found existing Pinecone index: {self.index_name}")
                try:
                    self.index = self.client.Index(self.index_name)
                    self.connected = True
                    logger.info("✅ Successfully connected to Pinecone index")
                    
                    # Try to get index stats to confirm connection is working
                    try:
                        stats = self.index.describe_index_stats()
                        vector_count = stats.get("total_vector_count", 0)
                        logger.info(f"Index contains {vector_count} vectors")
                    except Exception as e:
                        logger.warning(f"Could not get index stats: {e}")
                        
                except Exception as e:
                    logger.error(f"❌ Error connecting to index: {e}")
                    return
            else:
                logger.warning(f"❌ Index {self.index_name} does not exist and won't be created automatically")
                # Disabling automatic index creation as it could be causing startup issues
                self.connected = False
                
        except Exception as e:
            logger.error(f"❌ Error connecting to Pinecone: {e}")
            self.connected = False
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create an embedding for the given text using Anthropic.
        
        Args:
            text: The text to embed
            
        Returns:
            list: The embedding vector or None if failed
        """
        if not self.anthropic_client:
            logger.warning("❌ Anthropic client not available, cannot create embedding")
            return None
            
        try:
            # Truncate text if it's too long (Anthropic also has token limits)
            text_length = len(text)
            if text_length > 20000:
                logger.warning(f"Text too long ({text_length} chars), truncating to 20000 chars")
                text = text[:20000]
            
            logger.info(f"Creating embedding for text of length {len(text)} chars...")
            
            # Use Anthropic to generate a summary and then create a simple embedding
            # Since Anthropic doesn't have a direct embedding API, we'll use Claude to extract key features
            # and create a simplified embedding
            
            prompt = f"""Given the following text, extract the 10-15 most important keywords or phrases that represent the main topics and concepts. For each keyword, assign a relevance score from 0 to 1. Return the results as a JSON array of objects with 'keyword' and 'score' properties.

Text to analyze:
{text}

The JSON output should be formatted exactly like this:
[
  {{"keyword": "example keyword 1", "score": 0.95}},
  {{"keyword": "example keyword 2", "score": 0.85}}
]

Do not include any other text in your response, just the JSON data.
"""
            
            # Call Claude to extract features
            try:
                logger.info("Calling Anthropic API to extract key features from text...")
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0,
                    system="You analyze text and extract the most important keywords with relevance scores. Output only valid JSON.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract the JSON response
                json_text = response.content[0].text
                
                # Clean up the JSON text to ensure it's valid
                json_text = json_text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text.replace("```json", "", 1)
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                json_text = json_text.strip()
                
                # Parse the JSON
                try:
                    keywords = json.loads(json_text)
                    logger.info(f"✅ Successfully extracted {len(keywords)} keywords from text")
                    
                    # Convert to a simple vector
                    # We'll create a fixed-size vector of 1536 dimensions (same as OpenAI embeddings)
                    # each dimension representing a feature
                    
                    # Create a seed for consistent hashing
                    np.random.seed(42)
                    
                    # Initialize the embedding vector
                    embedding = np.zeros(1536)
                    
                    # For each keyword, create a random vector and multiply by its score
                    for item in keywords:
                        keyword = item.get("keyword", "")
                        score = item.get("score", 0.5)
                        
                        # Create a hash of the keyword
                        keyword_hash = int(hashlib.md5(keyword.encode()).hexdigest(), 16)
                        
                        # Use the hash to seed a random vector
                        np.random.seed(keyword_hash)
                        keyword_vector = np.random.normal(0, 1, 1536)
                        
                        # Add the weighted keyword vector to the embedding
                        embedding += keyword_vector * score
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    # Convert to list and return
                    embedding_list = embedding.tolist()
                    
                    logger.info(f"✅ Created embedding with {len(embedding_list)} dimensions")
                    return embedding_list
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Error parsing JSON response: {e}")
                    logger.error(f"Raw response: {json_text}")
                    return None
                
            except Exception as e:
                logger.error(f"❌ Error calling Anthropic: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating embedding: {e}")
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
            logger.info(f"Upserting vector with metadata: domain={domain}, class={metadata.get('predicted_class')}")
            
            # Upsert vector
            self.index.upsert(
                vectors=[(vector_id, embedding, sanitized_metadata)]
            )
            
            logger.info(f"✅ Successfully upserted vector for domain {domain} to Pinecone!")
            return True
        except Exception as e:
            logger.error(f"❌ Error upserting vector for {domain}: {e}")
            logger.error(traceback.format_exc())
            return False
