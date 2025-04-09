import os
import re
import logging
import pickle
import json
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DO NOT import scikit-learn at module level
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder

try:
    import nltk
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    HAS_STOPWORDS = True
except:
    STOPWORDS = set()
    HAS_STOPWORDS = False

class DomainClassifier:
    def __init__(
        self,
        model_path=None,
        use_pinecone=False,
        pinecone_api_key=None,
        pinecone_index_name=None,
        confidence_threshold=0.6,
        use_llm=False,
        anthropic_api_key=None,
        llm_model="claude-3-haiku-20240307"
    ):
        pinecone_api_key = pinecone_api_key or os.environ.get('PINECONE_API_KEY')
        anthropic_api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')

        # Check if API keys are present
        if use_pinecone and not pinecone_api_key:
            logger.error("Pinecone API key is missing")
            use_pinecone = False

        if use_llm and not anthropic_api_key:
            logger.error("Anthropic API key is missing")
            use_llm = False

        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.classes = ["Integrator - Commercial A/V", "Integrator - Residential A/V", "Managed Service Provider"]
        self.confidence_threshold = confidence_threshold
        self.use_pinecone = use_pinecone
        self.pinecone_index = None
        self._using_fallback = True

        # LLM integration
        self.use_llm = use_llm
        self.llm_classifier = None
        if use_llm and anthropic_api_key:
            # Defer import to avoid startup errors
            from llm_classifier import LLMClassifier
            self.llm_classifier = LLMClassifier(api_key=anthropic_api_key, model=llm_model)
            logger.info(f"Initialized LLM classifier with model: {llm_model}")

        if use_pinecone:
            try:
                # Updated Pinecone initialization
                from pinecone import Pinecone
                pc = Pinecone(api_key=pinecone_api_key)
                self.pinecone_index = pc.Index(pinecone_index_name)
                logger.info(f"Connected to Pinecone index: {pinecone_index_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                self.use_pinecone = False

        if model_path:
            self.load_model(model_path)

    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = text.split()
        if HAS_STOPWORDS:
            tokens = [token for token in tokens if token not in STOPWORDS]
        return ' '.join(tokens)

    def load_model(self, path):
        try:
            # Try to load the model but handle errors gracefully
            logger.info(f"Attempting to load model from {path}")
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier']
                self.label_encoder = data.get('label_encoder')
                self.classes = data['classes']
                self.confidence_threshold = data.get('confidence_threshold', 0.6)
                self._using_fallback = False
                logger.info(f"Model loaded successfully: {path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise e
        except Exception as e:
            logger.error(f"Using fallback classifier due to model loading error: {e}")
            self._using_fallback = True
            # No need to create a TfidfVectorizer - just use a dummy implementation

    def classify_domain(self, domain_content, domain=None):
        """
        Modified to prioritize LLM classification when the ML model fails to load
        """
        processed_text = self.preprocess_text(domain_content)
        
        if len(processed_text.split()) < 20:
            logger.warning(f"[{domain}] Content too short, skipping.")
            return {
                "predicted_class": "Unknown",
                "confidence_scores": {},
                "max_confidence": 0.0,
                "low_confidence": True,
                "detection_method": "content_length_check"
            }
            
        # If using fallback mode, rely on LLM classification
        if self._using_fallback or not self.classifier:
            logger.info(f"Using LLM-only classification for {domain}")
            if not self.use_llm or not self.llm_classifier:
                # If LLM is also not available, use a basic heuristic classifier
                logger.warning("Fallback ML model and LLM not available - using basic heuristic classification")
                return self._basic_heuristic_classify(domain_content, domain)
                
            # Get classification from LLM
            try:
                llm_result = self.llm_classifier.classify(domain_content, domain)
                logger.info(f"LLM classification for {domain}: {llm_result['predicted_class']} ({llm_result.get('max_confidence', 0.7):.2f})")
                
                # Add necessary fields for compatibility
                if "max_confidence" not in llm_result:
                    confidence_scores = llm_result.get("confidence_scores", {})
                    max_confidence = max(confidence_scores.values()) if confidence_scores else 0.7
                    llm_result["max_confidence"] = max_confidence
                    
                if "low_confidence" not in llm_result:
                    llm_result["low_confidence"] = llm_result["max_confidence"] < self.confidence_threshold
                
                llm_result["detection_method"] = "llm_classification_only"
                return llm_result
            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
                return self._basic_heuristic_classify(domain_content, domain)
        
        # This shouldn't be reached with the fallback, but keeping for completeness
        logger.warning("Using traditional ML classification (should not happen with fallback)")
        return self._basic_heuristic_classify(domain_content, domain)
    
    def _basic_heuristic_classify(self, content, domain=None):
        """Simple keyword-based classification when all else fails"""
        content = content.lower()
        
        # Simple keyword counting
        av_commercial_keywords = ['commercial', 'conference room', 'office', 'corporate', 'business', 'digital signage']
        av_residential_keywords = ['home theater', 'residential', 'home automation', 'smart home', 'living room']
        msp_keywords = ['managed service', 'it service', 'network', 'cloud', 'technical support', 'helpdesk', 'cyber', 'server']
        
        commercial_score = sum(1 for kw in av_commercial_keywords if kw in content)
        residential_score = sum(1 for kw in av_residential_keywords if kw in content)
        msp_score = sum(1 for kw in msp_keywords if kw in content)
        
        total = commercial_score + residential_score + msp_score
        if total == 0:
            # No keywords found
            return {
                "predicted_class": "Unknown",
                "confidence_scores": {
                    "Integrator - Commercial A/V": 0.33,
                    "Integrator - Residential A/V": 0.33,
                    "Managed Service Provider": 0.34
                },
                "max_confidence": 0.34,
                "low_confidence": True,
                "detection_method": "basic_heuristic_no_matches"
            }
        
        # Calculate normalized scores
        scores = {
            "Integrator - Commercial A/V": commercial_score / total,
            "Integrator - Residential A/V": residential_score / total,
            "Managed Service Provider": msp_score / total
        }
        
        max_class = max(scores.items(), key=lambda x: x[1])
        
        return {
            "predicted_class": max_class[0],
            "confidence_scores": scores,
            "max_confidence": max_class[1],
            "low_confidence": max_class[1] < self.confidence_threshold,
            "detection_method": "basic_heuristic_classification"
        }

    def get_embedding(self, domain):
        if not self.use_pinecone or not self.pinecone_index:
            return None
        try:
            result = self.pinecone_index.fetch(ids=[domain])
            if domain in result.vectors:
                return np.array(result.vectors[domain].values)
            return None
        except Exception as e:
            logger.error(f"Error fetching embedding for {domain}: {e}")
            return None

    def store_embedding(self, domain, embedding, metadata=None):
        if not self.use_pinecone or not self.pinecone_index:
            logger.warning("Pinecone not configured properly.")
            return False
        if embedding is None:
            logger.error(f"No embedding available to store for domain: {domain}")
            return False
        metadata = metadata or {}
        try:
            self.pinecone_index.upsert([(domain, embedding.tolist(), metadata)])
            logger.info(f"Embedding stored in Pinecone for domain: {domain}")
            return True
        except Exception as e:
            logger.error(f"Pinecone embedding storage error for {domain}: {e}")
            return False
