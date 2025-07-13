from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorConfig
from typing import Dict, Any
import hashlib

class OpenAIPayloadAnonymizer:
    def __init__(self, salt: str):
        self.salt = salt
        # 1. Define the configuration dictionary
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }

        # 2. Use the NlpEngineProvider to create the NLP engine object
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # 3. Pass the created NLP engine object to the AnalyzerEngine
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()
        
        # Custom operators for consistent anonymization
        self.operators = {
            "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": self._consistent_hash}),
            "PHONE_NUMBER": OperatorConfig("custom", {"lambda": self._consistent_hash}),
            "PERSON": OperatorConfig("custom", {"lambda": self._consistent_hash}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
        }
    
    def _consistent_hash(self, text: str) -> str:
        """Generate consistent hashes for the same input"""
        return f"anon_{hashlib.sha256((text + self.salt).encode()).hexdigest()[:8]}"
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize a single text string"""
        results = self.analyzer.analyze(text=text, language="en")
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self.operators
        )
        return anonymized.text
    
    def anonymize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process OpenAI API payload"""
        if "messages" in payload:
            for message in payload["messages"]:
                if "content" in message and isinstance(message["content"], str):
                    message["content"] = self.anonymize_text(message["content"])
        
        # Anonymize other potential PII fields
        for field in ["user", "session_id"]:
            if field in payload and isinstance(payload[field], str):
                payload[field] = self.anonymize_text(payload[field])
        
        return payload