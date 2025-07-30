from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine, EngineResult, OperatorConfig, DeanonymizeEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorResult
from typing import Dict, Any, List, cast

from .InstanceCounterAnonymizer import InstanceCounterAnonymizer
from .InstanceCounterDeanonymizer import InstanceCounterDeanonymizer


class OpenAIPayloadAnonymizer:
    def __init__(self):
        # NLP setup
        configuration : dict[str, Any] = {
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
                    "EMAIL": "EMAIL",
                    "PHONE_NUMBER": "PHONE_NUMBER",
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

        # Add custom recognizers for specific PII patterns
        self._add_custom_recognizers()

        self.anonymizerEngine = AnonymizerEngine()
        self.anonymizerEngine.add_anonymizer( InstanceCounterAnonymizer )

        # Create a mapping between entity types and counters
        self.entity_mapping = dict[str, int]()

        self.deanonymizer_engine = DeanonymizeEngine()

        self.deanonymizer_engine.add_deanonymizer( InstanceCounterDeanonymizer )

        # Mapping for reversible anonymization
        self.forward_map: Dict[str, str] = {}
        self.reverse_map: Dict[str, str] = {}

        # Counters per entity type
        self.entity_counters: Dict[str, int] = {}

    def _add_custom_recognizers(self):
        """Add custom pattern recognizers for specific PII types"""
        custom_recognizers = [
        # Username recognizer (e.g., user123, admin_456)
        PatternRecognizer(
            supported_entity="USERNAME",
            deny_list=[],
            patterns=[
                Pattern(
                    name="username_pattern",  # Descriptive name
                    regex=r"\b(?=\w*[a-zA-Z])(?=\w*\d)\w{5,}\b",  # Your regex
                    score=0.9  # Confidence score (0-1)
                )
            ],
            context=["user", "login", "username", "handle", "account"],
            supported_language="en"
        ),
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            deny_list=[],  # You can add specific numbers to deny if needed
            patterns=[
                Pattern(
                        name="flexible_phone_number_pattern",
                        # This regex aims to match common US phone number formats, including 7-digit ones:
                        # (123) 456-7890
                        # 123-456-7890
                        # 123.456.7890
                        # 123 456 7890
                        # +1 123-456-7890 (optional country code)
                        # 555-1234 (7-digit format)
                        # 5551234 (7-digit format without hyphen)
                        regex=r"\b(?:\+?\d{1,3}[-. ]?)?(?:\(?\d{3}\)?[-. ]?)?\d{3}[-. ]?\d{4}\b",
                        score=0.9
                    )
            ],
            context=["phone", "contact", "mobile", "call"],
            supported_language="en"
        )
        ]
        for recognizer in custom_recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

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

    def anonymize_text(self, text: str) -> EngineResult:
        """Anonymize and label PII in text"""
        analyzer_results : List[RecognizerResult] = self.analyzer.analyze(text=text, language="en", score_threshold=0.6)

        ordered_values: Dict[str, list[str]] = {}

        self.order_entities_in_order_of_appearence(text, analyzer_results, ordered_values)

        anonymized_result = self.anonymizerEngine.anonymize(
            text=text,
            analyzer_results=analyzer_results,  # type: ignore
            operators={
                "DEFAULT": OperatorConfig(
                    "entity_counter",
                    {
                        "entity_mapping": self.entity_mapping,
                        "ordered_values": ordered_values
                    }
                )
            },
        )
        # new_text = anonymized_result.text

        return anonymized_result

    # the method below is used to order entities in the order they appear in the text
    # so that in the text "Alice and Bob are friends" will be anonymized as
    # "<PERSON_0> and <PERSON_1> are friends" and not "<PERSON_1> and <PERSON_0> are friends"
    def order_entities_in_order_of_appearence(self, text, analyzer_results, ordered_values):
        for r in analyzer_results:
            r_text = text[r.start:r.end]
            entity_type = r.entity_type
            if entity_type not in ordered_values:
                ordered_values[entity_type] = []
            if r_text not in ordered_values[entity_type]:
                ordered_values[entity_type].append(r_text)

    def deanonymize_text(self, text: str, operator_results: List[OperatorResult]) -> str:
        """Replace placeholders like <PERSON_1> with original values"""
        # def replace_match(match):
        #     token = match.group(0)
        #     return self.reverse_map.get(token, token)

        # return re.sub(r"<[A-Z_]+_\d+>", replace_match, text)
        anonymized_result = self.deanonymizer_engine.deanonymize(
            text=text,
            entities=operator_results,
            operators={
                "DEFAULT": OperatorConfig(
                    "entity_counter_deanonymizer", {"entity_mapping": self.entity_mapping}
                )
            },
        )
        return anonymized_result.text

    def anonymize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively anonymize all string leaf values in a JSON-like payload"""
        def recursive_anonymize(obj: Any) -> Any:
            if isinstance(obj, dict):
                obj_dict = cast(dict[Any, Any], obj)
                return {k: recursive_anonymize(v) for k, v in obj_dict.items()}
            elif isinstance(obj, list):
                obj_list = cast(list[Any], obj)
                return [recursive_anonymize(item) for item in obj_list]
            elif isinstance(obj, str):
                # Assuming anonymize_text() returns an object with a .text attribute
                return self.anonymize_text(obj).text
            else:
                return obj
        return recursive_anonymize(payload)

    def deanonymize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._create_reverse_map()
        # Recursively restore original values in payload using reverse_map
        return self._recursive_deanonymize(payload)
    
    def _create_reverse_map(self):
        # Create reverse mapping (anonymized_token → real_value)
        entity_mapping = cast(Dict[str, Dict[str, str]], self.entity_mapping)
        self.reverse_map: Dict[str, str] = {}
        for _entity_type, mappings in entity_mapping.items():
            for real_value, anonymized_token in mappings.items():
                self.reverse_map[anonymized_token] = real_value

    def _recursive_deanonymize(self, obj: Any) -> Any:
        """Helper to recursively deanonymize strings in dicts/lists"""
        if isinstance(obj, dict):
            obj_dict = cast(dict[Any, Any], obj)
            return { key: self._recursive_deanonymize(value) for key, value in obj_dict.items() }
        elif isinstance(obj, list):
            obj_list = cast(list[Any], obj)
            return [self._recursive_deanonymize(item) for item in obj_list]
        elif isinstance(obj, str):
            return self._deanonymize_string(obj)
        else:
            return obj

    def _deanonymize_string(self, text: str) -> str:
        """Replace all anonymized tokens in a string using reverse_map"""
        # deanonymized: str = self.deanonymize_text(s, [])
        # Replace each anonymized token with its original value
        for anonymized_token, real_value in self.reverse_map.items():
            text = text.replace(anonymized_token, real_value)
        
        return text
