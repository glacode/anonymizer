import copy
from typing import Any, Dict
import pytest
from openai_anonymizer.anonymizer import OpenAIPayloadAnonymizer

class TestOpenAIPayloadAnonymizer:
    @pytest.fixture
    def anonymizer(self):
        """Fixture providing a fresh anonymizer for each test"""
        return OpenAIPayloadAnonymizer()

    def test_initialization(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test that the anonymizer initializes correctly"""
        # assert hasattr(anonymizer, 'analyzer')
        # assert hasattr(anonymizer, 'anonymizer')
        # assert anonymizer.forward_map == {}
        # assert anonymizer.reverse_map == {}
        # assert anonymizer.entity_counters == {}
        assert anonymizer.analyzer is not None

    def test_anonymize_text_simple(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test basic text anonymization"""
        text = "My name is John Doe and I live in New York."
        anonymizedResult = anonymizer.anonymize_text(text)
        
        # Check that PII was replaced
        assert "John Doe" not in anonymizedResult.text
        assert "New York" not in anonymizedResult.text
        assert "<PERSON_" in anonymizedResult.text
        assert "<LOCATION_" in anonymizedResult.text
        
        # Check that non-PII remains
        assert "My name is" in anonymizedResult.text
        assert "and I live in" in anonymizedResult.text

    def test_deanonymize_text(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test text deanonymization"""
        original = "My name is John Doe and I live in New York."
        anonymized = anonymizer.anonymize_text(original)
        deanonymized = anonymizer.deanonymize_text(
            anonymized.text,
            operator_results=anonymized.items
        )

        assert deanonymized == original

    def test_anonymize_multiple_same_entities(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test that the same entity gets the same anonymization"""
        text1 = "My name is John Doe."
        text2 = "John Doe is my name."
        
        anonymizer.anonymize_text(text1)
        anonymized2 = anonymizer.anonymize_text(text2)
        
        # Check the same tag was used in both texts
        assert "<PERSON_0>" in anonymized2.text

    def test_anonymize_payload_messages(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test payload message anonymization"""
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "My name is John Doe"},
                {"role": "assistant", "content": "Hello John Doe"}
            ],
            "user": "user123",
            "other_field": "unchanged"
        }
        
        anonymized = anonymizer.anonymize_payload(payload)
        
        # Check messages were anonymized
        anonymized0 : str = anonymized["messages"][0]["content"]
        anonymized1 : str = anonymized["messages"][1]["content"]
        assert "John Doe" not in anonymized0
        assert "John Doe" not in anonymized1
        assert "<PERSON_" in anonymized0
        assert "<PERSON_" in anonymized1

        # Check user field was anonymized
        anonymized_user: str = anonymized["user"]
        assert "user123" not in anonymized_user
        assert anonymized_user == "<USERNAME_0>"

        # Check other fields unchanged is not even anonymized
        assert isinstance(anonymized["other_field"], str)
        assert anonymized["other_field"] == "unchanged"

    def test_deanonymize_payload(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test payload deanonymization"""
        original_payload : Dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "My name is John Doe"},
                {"role": "assistant", "content": "Hello John Doe"}
            ],
            "user": "user123"
        }
        
        deep_copy = copy.deepcopy(original_payload)
        
        # First anonymize
        anonymized = anonymizer.anonymize_payload(deep_copy)
        
        # Then deanonymize
        deanonymized = anonymizer.deanonymize_payload(anonymized)
        
        # Check it matches original
        assert original_payload["messages"]== deanonymized["messages"]
        assert original_payload["user"] == deanonymized["user"]
        assert deanonymized == original_payload

    def test_empty_text(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test handling of empty text"""
        assert anonymizer.anonymize_text("").text == ""
        assert anonymizer.deanonymize_text("",[]) == ""

    def test_no_pii_text(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test text with no PII"""
        text = "This is just a normal sentence."
        anonymized_result = anonymizer.anonymize_text(text)
        assert anonymized_result.text == text
        assert anonymizer.deanonymize_text(text,anonymized_result.items) == text

    def test_multiple_entity_types(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test handling of multiple entity types"""
        text = "John Doe lives in New York and works at Google."
        anonymized = anonymizer.anonymize_text(text)

        assert "<PERSON_" in anonymized.text
        assert "<LOCATION_" in anonymized.text
        assert "<ORG_" in anonymized.text

        deanonymized = anonymizer.deanonymize_text(anonymized.text, anonymized.items)
        assert deanonymized == text

    def test_partial_payload(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test payloads with missing fields"""
        payload = {"messages": [{"role": "user"}]}  # No content
        assert anonymizer.anonymize_payload(payload) == payload
        
        payload = {}  # Empty payload
        assert anonymizer.anonymize_payload(payload) == payload

    def test_non_string_fields(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test payload with non-string fields"""
        payload = {
            "messages": [{"role": "user", "content": 123}],  # Numeric content
            "user": None,  # None value
            "flag": True  # Boolean value
        }
        
        # Should not raise exceptions
        anonymized = anonymizer.anonymize_payload(payload)
        assert anonymized == payload

    def test_reversible_anonymization(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test that anonymization is fully reversible"""
        original: Dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "My phone is 555-1234"},
                {"role": "assistant", "content": "I'll call 555-1234"}
            ],
            "user": "test@example.com",
            "session_id": "session123"
        }

        deep_copy = copy.deepcopy(original)

        anonymized = anonymizer.anonymize_payload(deep_copy)
        deanonymized = anonymizer.deanonymize_payload(anonymized)
        
        assert deanonymized == original

    def test_multiple_calls_maintain_state(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test that multiple calls maintain proper mapping state"""
        text1 = "Contact Bob at bob@example.com"
        text2 = "Bob's number is 555-1234"
        
        # First call
        anonymized1 = anonymizer.anonymize_text(text1)
        print(anonymizer.analyzer.analyze(text=text1, language="en"))
        assert "<PERSON_" in anonymized1.text
        assert "<EMAIL_ADDRESS_" in anonymized1.text
        assert anonymized1.text == "Contact <PERSON_0> at <EMAIL_ADDRESS_0>"

        # Second call
        anonymized2 = anonymizer.anonymize_text(text2)
        assert "<PERSON_" in anonymized2.text
        assert "<PHONE_NUMBER_" in anonymized2.text
        assert anonymized2.text == "<PERSON_0>'s number is <PHONE_NUMBER_0>"


        # Check same person tag was used
        person_tag = [word for word in anonymized1.text.split() if word.startswith("<PERSON_")][0]
        assert person_tag in anonymized2.text

        # Check deanonymization works
        deanonymized1 = anonymizer.deanonymize_text(anonymized1.text, anonymized1.items)
        deanonymized2 = anonymizer.deanonymize_text(anonymized2.text, anonymized2.items)

        assert deanonymized1 == text1
        assert deanonymized2 == text2

    def test_multiple_entities_of_the_same_type(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test that multiple entities of the same type are handled correctly"""
        text = "Alice and Bob are friends. They both work at Acme Corp."
        anonymized = anonymizer.anonymize_text(text)

        assert "<PERSON_" in anonymized.text
        assert "<ORG_" in anonymized.text
        assert anonymized.text == "<PERSON_0> and <PERSON_1> are friends. They both work at <ORG_0>"

        deanonymized = anonymizer.deanonymize_text(anonymized.text, anonymized.items)
        assert deanonymized == text
        assert deanonymized == "Alice and Bob are friends. They both work at Acme Corp."
    
    def test_random_secret_anonymization(self, anonymizer: OpenAIPayloadAnonymizer):
        """Test anonymization of moderate-entropy secrets using RANDOM_SECRET recognizer"""
        # Example that should be detected: includes digit + symbol, and is at least 8 chars
        original = "My token is: 1!23456A and some text follow"
        
        anonymized = anonymizer.anonymize_text(original)

        # Check that the secret was replaced
        assert "<RANDOM_SECRET_0>" in anonymized.text, "Secret not anonymized"
        assert "1!23456A" not in anonymized.text, "Original secret still present"
        assert anonymized.text == "My token is: <RANDOM_SECRET_0> and some text follow"

        # Test deanonymization
        deanonymized = anonymizer.deanonymize_text(
            anonymized.text,
            operator_results=anonymized.items
        )
        assert deanonymized == original, "Deanonymization failed"

    @pytest.mark.parametrize("secret", [
        "1!23456A",        # digit + symbol + upper
        "hunter2!",        # lower + digit + symbol
        "P@ssword",        # upper + lower + symbol
        "sk-123abcXYZ",    # typical API key
        "Zx8#Fg7*",        # symbol-heavy
    ])
    def test_random_secret_parametrized(self, anonymizer: OpenAIPayloadAnonymizer, secret: str):
        """Ensure multiple random-like secrets are caught by RANDOM_SECRET"""
        original = f"My secret is: {secret}"
        
        anonymized = anonymizer.anonymize_text(original)
        print("Anonymized:", anonymized.text)

        assert "<RANDOM_SECRET_0>" in anonymized.text
        assert secret not in anonymized.text

        # Optional: check round-trip
        deanonymized = anonymizer.deanonymize_text(
            anonymized.text,
            operator_results=anonymized.items
        )
        assert deanonymized == original

    @pytest.mark.parametrize("secret", [
        "1!23456A",        # digit + symbol + upper
        "hunter2!",        # lower + digit + symbol
        "P@ssword",        # upper + lower + symbol
        "sk-123abcXYZ",    # typical API key
        "Zx8#Fg7*",        # symbol-heavy
    ])
    def test_random_secret_in_the_middle_parametrized(self, anonymizer: OpenAIPayloadAnonymizer, secret: str):
        """Ensure multiple random-like secrets are caught by RANDOM_SECRET"""
        original = f"My secret is: {secret} ; but some text follows"
        
        anonymized = anonymizer.anonymize_text(original)
        print("Anonymized:", anonymized.text)

        assert "<RANDOM_SECRET_0>" in anonymized.text
        assert secret not in anonymized.text

        # Optional: check round-trip
        deanonymized = anonymizer.deanonymize_text(
            anonymized.text,
            operator_results=anonymized.items
        )
        assert deanonymized == original
