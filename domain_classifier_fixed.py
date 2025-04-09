# domain_classifier_fixed.py - Create this as a new file
import os
import re
import logging
import json
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NO scikit-learn imports whatsoever!

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
        self.confidence_threshold = confidence_threshold
        self.classes = ["Integrator - Commercial A/V", "Integrator - Residential A/V", "Managed Service Provider"]
        self._using_fallback = True
        
        # LLM integration
        self.use_llm = use_llm
        self.llm_classifier = None
        if use_llm and anthropic_api_key:
            try:
                from llm_classifier import LLMClassifier
                self.llm_classifier = LLMClassifier(api_key=anthropic_api_key, model=llm_model)
                logger.info(f"Initialized LLM classifier with model: {llm_model}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM classifier: {e}")
        
        # Pinecone integration
        self.use_pinecone = use_pinecone
        self.pinecone_index = None
        if use_pinecone and pinecone_api_key:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=pinecone_api_key)
                self.pinecone_index = pc.Index(pinecone_index_name)
                logger.info(f"Connected to Pinecone index: {pinecone_index_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                self.use_pinecone = False
        
        logger.info("Using LLM-only classification mode")

    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return ' '.join(text.split())

    def classify_domain(self, domain_content, domain=None):
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
        
        # Try LLM classification
        if self.use_llm and self.llm_classifier:
            try:
                llm_result = self.llm_classifier.classify(domain_content, domain)
                logger.info(f"LLM classification for {domain}: {llm_result.get('predicted_class')}")
                
                # Add necessary fields for compatibility
                if "max_confidence" not in llm_result and "confidence_scores" in llm_result:
                    confidence_scores = llm_result.get("confidence_scores", {})
                    max_confidence = max(confidence_scores.values()) if confidence_scores else 0.7
                    llm_result["max_confidence"] = max_confidence
                elif "max_confidence" not in llm_result:
                    llm_result["max_confidence"] = 0.7
                    
                if "low_confidence" not in llm_result:
                    llm_result["low_confidence"] = llm_result["max_confidence"] < self.confidence_threshold
                
                if "detection_method" not in llm_result:
                    llm_result["detection_method"] = "llm_classification_only"
                
                return llm_result
            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
        
        # Fallback to basic heuristic
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
