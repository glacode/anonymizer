from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorResult
from typing import Dict, Any
import re


class OpenAIPayloadAnonymizer:
    def __init__(self):
        # NLP setup
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en_core_web_sm"
                    # "model_name": "en_core_web_lg"
                }
            ],
            "ner_model_configuration": {
                # Detected entities for the en_core_web_sm model are here https://spacy.io/models/en#en_core_web_sm
                # are listed in ./listEntities.py (and are the keys in the dict below, left side of the colon)
                # PII entities supported by Presidio are here https://microsoft.github.io/presidio/supported_entities/ 
                # they are the values (right side of the colon) in the dict below
                "model_to_presidio_entity_mapping": {
                    "PERSON": "PERSON",
                    "GPE": "LOCATION",
                    "ORG": "ORG",
                    # Add other mappings as needed
                },
                "low_score_entity_names": [],  # List of entities to ignore if low confidence
                "labels_to_ignore": []         # List of labels to completely ignore
            }
            # "labels_to_ignore": []
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en","es","it","pl"],  # or whichever languages you actually need
            #deny_list=["CreditCardRecognizer"]  # optional: if you don't need this recognizer
        )
        self.anonymizer = AnonymizerEngine()

        # Mapping for reversible anonymization
        self.forward_map: Dict[str, str] = {}
        self.reverse_map: Dict[str, str] = {}

        # Counters per entity type
        self.entity_counters: Dict[str, int] = {}

    def _get_label(self, entity_type: str) -> str:
        """Generate sequential anonymized labels like <PERSON_1>"""
        self.entity_counters.setdefault(entity_type, 0)
        self.entity_counters[entity_type] += 1
        return f"<{entity_type}_{self.entity_counters[entity_type]}>"

    def _anonymize_entity(self, text: str, entity_type: str) -> str:
        """Return consistent label for a given text"""
        if text in self.forward_map:
            return self.forward_map[text]

        label = self._get_label(entity_type)
        self.forward_map[text] = label
        self.reverse_map[label] = text
        return label

    def anonymize_text(self, text: str) -> str:
        """Anonymize and label PII in text"""
        results = self.analyzer.analyze(text=text, language="en")
        new_text = text
        offset_correction = 0

        for result in sorted(results, key=lambda r: r.start):
            original_value = text[result.start:result.end]
            label = self._anonymize_entity(original_value, result.entity_type)

            # Replace original value with label (accounting for previous replacements)
            start = result.start + offset_correction
            end = result.end + offset_correction
            new_text = new_text[:start] + label + new_text[end:]
            offset_correction += len(label) - (result.end - result.start)

        return new_text

    def deanonymize_text(self, text: str) -> str:
        """Replace placeholders like <PERSON_1> with original values"""
        def replace_match(match):
            token = match.group(0)
            return self.reverse_map.get(token, token)

        return re.sub(r"<[A-Z_]+_\d+>", replace_match, text)

    def anonymize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize content in OpenAI API request"""
        if "messages" in payload:
            for message in payload["messages"]:
                if "content" in message and isinstance(message["content"], str):
                    message["content"] = self.anonymize_text(message["content"])

        for field in ["user", "session_id"]:
            if field in payload and isinstance(payload[field], str):
                payload[field] = self.anonymize_text(payload[field])

        return payload

    def deanonymize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Restore original content from anonymized OpenAI API response"""
        if "choices" in payload:
            for choice in payload["choices"]:
                message = choice.get("message", {})
                if "content" in message and isinstance(message["content"], str):
                    message["content"] = self.deanonymize_text(message["content"])

        return payload
