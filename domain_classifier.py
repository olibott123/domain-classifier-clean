import os
import re
import logging
import pickle
import numpy as np
import pinecone
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from llm_classifier import LLMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.classes = None
        self.confidence_threshold = confidence_threshold
        self.use_pinecone = use_pinecone
        self.pinecone_index = None
        
        # LLM integration
        self.use_llm = use_llm
        self.llm_classifier = None
        if use_llm and anthropic_api_key:
            self.llm_classifier = LLMClassifier(api_key=anthropic_api_key, model=llm_model)
            logger.info(f"Initialized LLM classifier with model: {llm_model}")

        if use_pinecone:
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            self.pinecone_index = pc.Index(pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {pinecone_index_name}")

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

    def classify_domain(self, domain_content, domain=None):
        if not self.classifier or not self.vectorizer:
            raise ValueError("Classifier not trained or loaded.")

        processed_text = self.preprocess_text(domain_content)

        if len(processed_text.split()) < 20:
            logger.warning(f"[{domain}] Content too short, skipping.")
            return {
                "predicted_class": "Unknown",
                "confidence_scores": {},
                "low_confidence": True,
                "detection_method": "content_length_check"
            }

        # Step 1: Always check and manage Pinecone embeddings for traditional model
        embedding = None
        embedding_exists = False

        if domain and self.use_pinecone:
            embedding = self.get_embedding(domain)
            embedding_exists = embedding is not None

        # If embedding not found in Pinecone, generate it and store it
        if not embedding_exists:
            X = self.vectorizer.transform([processed_text])
            embedding = X.toarray()[0]

            if embedding.size > 0:
                stored_successfully = self.store_embedding(domain, embedding, metadata={
                    "domain": domain,
                    "content_length": len(domain_content),
                    "classification_date": datetime.now().isoformat()
                })
                if stored_successfully:
                    logger.info(f"Embedding stored in Pinecone for domain: {domain}")
                else:
                    logger.error(f"Failed to store embedding for domain: {domain}")
            else:
                logger.error(f"Generated embedding invalid for domain {domain}.")
        else:
            logger.info(f"Using existing Pinecone embedding for domain: {domain}")

        # Step 2: Get classification from traditional model
        probabilities = self.classifier.predict_proba(embedding.reshape(1, -1))[0]
        traditional_scores = {cls: float(prob) for cls, prob in zip(self.classes, probabilities)}
        traditional_class = self.classes[np.argmax(probabilities)]
        traditional_confidence = max(probabilities)
        
        # If we're not using LLM classification, just return traditional results
        if not self.use_llm or not self.llm_classifier:
            logger.info(f"Traditional classification for {domain}: {traditional_class} ({traditional_confidence:.2f})")
            return {
                "predicted_class": traditional_class,
                "confidence_scores": traditional_scores,
                "max_confidence": float(traditional_confidence),
                "low_confidence": traditional_confidence < self.confidence_threshold,
                "detection_method": "traditional_ml_classification"
            }
        
        # Step 3: Get classification from LLM
        try:
            llm_result = self.llm_classifier.classify(domain_content, domain)
            logger.info(f"LLM classification for {domain}: {llm_result['predicted_class']} "
                      f"({llm_result['max_confidence']:.2f}) - {llm_result['explanation'][:100]}...")
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            llm_result = {
                "predicted_class": traditional_class,
                "confidence_scores": traditional_scores,
                "max_confidence": 0.0,
                "explanation": f"LLM classification failed: {e}"
            }
        
        # Step 4: Combine traditional and LLM results
        final_result = self._combine_classifications(traditional_class, traditional_confidence, 
                                                  traditional_scores, llm_result, domain)
        
        # Step 5: Check for internal IT department if confidence is low
        if final_result['low_confidence'] or max(final_result['confidence_scores'].values()) < 0.7:
            logger.info(f"Low confidence for standard classifications, checking if {domain} is an internal IT department")
            
            # Check if it's an internal IT department
            it_dept_result = self.llm_classifier.detect_internal_it_department(domain_content, domain)
            
            if it_dept_result['is_internal_it'] and it_dept_result['confidence'] > 0.6:
                # Override classification to Internal IT Department
                final_result['predicted_class'] = "Internal IT Department"
                final_result['detection_method'] = "internal_it_detection"
                final_result['max_confidence'] = it_dept_result['confidence']
                final_result['low_confidence'] = it_dept_result['confidence'] < 0.7
                final_result['it_department_explanation'] = it_dept_result['explanation']
                
                logger.info(f"Reclassified {domain} as Internal IT Department ({it_dept_result['confidence']:.2f})")
            else:
                logger.info(f"Not an internal IT department: {it_dept_result['explanation'][:100]}...")
        
        logger.info(f"Final classification for {domain}: {final_result['predicted_class']} "
                  f"({final_result['max_confidence']:.2f}) via {final_result['detection_method']}")
        
        return final_result

    def _combine_classifications(self, trad_class, trad_confidence, trad_scores, llm_result, domain):
        """Combine traditional ML and LLM classification results"""
        llm_class = llm_result["predicted_class"]
        llm_confidence = llm_result["max_confidence"]
        llm_explanation = llm_result.get("explanation", "No explanation provided")
        
        # Case 1: Models agree - use the classification with boosted confidence
        if trad_class == llm_class:
            # Agreement is a strong signal - boost confidence
            final_confidence = min(0.95, (trad_confidence + llm_confidence) / 2 + 0.1)
            detection_method = "traditional_llm_agreement"
            final_class = trad_class
        
        # Case 2: LLM has high confidence - prefer it
        elif llm_confidence > 0.8:
            final_confidence = llm_confidence
            detection_method = "llm_high_confidence_override"
            final_class = llm_class
        
        # Case 3: Traditional has high confidence - prefer it
        elif trad_confidence > 0.75:
            final_confidence = trad_confidence
            detection_method = "traditional_high_confidence"
            final_class = trad_class
            
        # Case 4: Neither has high confidence - weighted average with slight LLM preference
        else:
            # When uncertain, give LLM slightly more weight
            llm_weight = 0.6
            trad_weight = 0.4
            
            # Average the confidences with weighting
            if llm_confidence > trad_confidence:
                final_class = llm_class
                final_confidence = (llm_confidence * llm_weight + trad_confidence * trad_weight)
                detection_method = "weighted_llm_preference"
            else:
                final_class = trad_class
                final_confidence = (trad_confidence * trad_weight + llm_confidence * llm_weight)
                detection_method = "weighted_traditional_preference"
        
        # Combine and normalize confidence scores
        # Start with traditional scores
        combined_scores = trad_scores.copy()
        
        # Add LLM scores with weighting
        for cls, score in llm_result.get("confidence_scores", {}).items():
            if cls in combined_scores:
                # Weighted average of scores
                combined_scores[cls] = (combined_scores[cls] + score) / 2
            else:
                combined_scores[cls] = score
                
        # Ensure the final class has the highest confidence
        combined_scores[final_class] = final_confidence
        
        return {
            "predicted_class": final_class,
            "confidence_scores": combined_scores,
            "max_confidence": float(final_confidence),
            "low_confidence": final_confidence < self.confidence_threshold,
            "detection_method": detection_method,
            "llm_explanation": llm_explanation
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

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.classes = data['classes']
        self.confidence_threshold = data.get('confidence_threshold', 0.6)
        logger.info(f"Model loaded: {path}")

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'classes': self.classes,
                'confidence_threshold': self.confidence_threshold
            }, f)
        logger.info(f"Model saved: {path}")
