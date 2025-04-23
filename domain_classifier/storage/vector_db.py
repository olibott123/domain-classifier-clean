"""Pinecone vector database connector for domain classification storage."""
import logging
import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import traceback

# Set up logging
logger = logging.getLogger(__name__)

try:
    # Import Pinecone
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    from pinecone import Config, PodSpec
    
    # Import embedding model
    import openai
    
    PINECONE_AVAILABLE = True
except ImportError:
    logger.warning("Pinecone or OpenAI not available, vector storage will be disabled")
    PINECONE_AVAILABLE = False

class VectorDBConnector:
    def __init__(self, 
                 api_key: str = None, 
                 index_name: str = None,
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the Pinecone vector database connector.
        
        Args:
            api_key: The API key for Pinecone
            index_name: The name of the Pinecone index to use
            embedding_model: The OpenAI model to use for embeddings
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "domain-classification")
        self.embedding_model = embedding_model
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.connected = False
        self.client = None
        self.index = None
        
        # Set up OpenAI client if API key is available
        if self.openai_api_key:
            try:
                openai.api_key = self.openai_api_key
                # Using new OpenAI client
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            logger.warning("No OpenAI API key provided, embeddings will not be available")
            self.openai_client = None
        
        # Initialize connection to Pinecone if API key is available
        if self.api_key and PINECONE_AVAILABLE:
            try:
                self._init_connection()
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone connection: {e}")
                self.connected = False
        else:
            if not self.api_key:
                logger.warning("No Pinecone API key provided, vector storage will not be available")
            self.connected = False
    
    def _init_connection(self):
        """Initialize the connection to Pinecone."""
        try:
            # Initialize Pinecone client
            self.client = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            existing_indexes = [index.name for index in self.client.list_indexes()]
            
            if self.index_name in existing_indexes:
                logger.info(f"Found existing Pinecone index: {self.index_name}")
                self.index = self.client.Index(self.index_name)
                self.connected = True
            else:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                # Create index with 1536 dimensions (OpenAI embedding dimension)
                try:
                    # First try serverless (preferred for new deployments)
                    self.client.create_index(
                        name=self.index_name,
                        dimension=1536,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-west-2"
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to create serverless index, falling back to pod-based: {e}")
                    # Fallback to pod-based
                    self.client.create_index(
                        name=self.index_name, 
                        dimension=1536,
                        metric="cosine",
                        spec=PodSpec(
                            environment="gcp-starter"
                        )
                    )
                
                # Connect to the newly created index
                self.index = self.client.Index(self.index_name)
                self.connected = True
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            self.connected = False
            raise
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create an embedding for the given text using OpenAI.
        
        Args:
            text: The text to embed
            
        Returns:
            list: The embedding vector or None if failed
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available, cannot create embedding")
            return None
            
        try:
            # Truncate text if it's too long
            if len(text) > 25000:  # OpenAI has token limits
                logger.warning(f"Text too long ({len(text)} chars), truncating to 25000 chars")
                text = text[:25000]
            
            # Create embedding
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            
            logger.info(f"Created embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
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
            logger.warning(f"Not connected to Pinecone, cannot upsert vector for {domain}")
            return False
            
        try:
            # Create embedding
            embedding = self.create_embedding(content)
            if not embedding:
                logger.warning(f"Failed to create embedding for {domain}")
                return False
            
            # Generate ID
            vector_id = self.generate_vector_id(domain)
            
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
            
            # Upsert vector
            self.index.upsert(
                vectors=[(vector_id, embedding, sanitized_metadata)]
            )
            
            logger.info(f"Upserted vector for domain {domain}")
            return True
        except Exception as e:
            logger.error(f"Error upserting vector for {domain}: {e}")
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
            logger.warning("Not connected to Pinecone, cannot query similar domains")
            return []
            
        try:
            # Create embedding
            embedding = self.create_embedding(query_text)
            if not embedding:
                logger.warning("Failed to create embedding for query")
                return []
            
            # Query Pinecone
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            
            # Process results
            similar_domains = []
            for match in results.matches:
                # Extract metadata
                metadata = match.metadata or {}
                domain = metadata.get("domain", "unknown")
                
                # Create result object
                result = {
                    "domain": domain,
                    "score": match.score,
                    "metadata": metadata
                }
                
                similar_domains.append(result)
            
            logger.info(f"Found {len(similar_domains)} similar domains")
            return similar_domains
        except Exception as e:
            logger.error(f"Error querying similar domains: {e}")
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
            logger.warning(f"Not connected to Pinecone, cannot delete vector for {domain}")
            return False
            
        try:
            # Generate ID
            vector_id = self.generate_vector_id(domain)
            
            # Delete vector
            self.index.delete(ids=[vector_id])
            
            logger.info(f"Deleted vector for domain {domain}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector for {domain}: {e}")
            return False
