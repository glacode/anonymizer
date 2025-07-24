import pytest
from openai_anonymizer.anonymizer import OpenAIPayloadAnonymizer

class TestOpenAIPayloadAnonymizer:
    @pytest.fixture
    def anonymizer(self):
        """Fixture providing a fresh anonymizer for each test"""
        return OpenAIPayloadAnonymizer()

    def test_initialization(self, anonymizer):
        """Test that the anonymizer initializes correctly"""
        assert hasattr(anonymizer, 'analyzer')
        assert hasattr(anonymizer, 'anonymizer')
        assert anonymizer.forward_map == {}
        assert anonymizer.reverse_map == {}
        assert anonymizer.entity_counters == {}

    def test_anonymize_text_simple(self, anonymizer):
        """Test basic text anonymization"""
        text = "My name is John Doe and I live in New York."
        anonymized = anonymizer.anonymize_text(text)
        
        # Check that PII was replaced
        assert "John Doe" not in anonymized
        assert "New York" not in anonymized
        assert "<PERSON_" in anonymized
        assert "<LOCATION_" in anonymized
        
        # Check that non-PII remains
        assert "My name is" in anonymized
        assert "and I live in" in anonymized

    def test_deanonymize_text(self, anonymizer):
        """Test text deanonymization"""
        original = "My name is John Doe and I live in New York."
        anonymized = anonymizer.anonymize_text(original)
        deanonymized = anonymizer.deanonymize_text(anonymized)
        
        assert deanonymized == original

    def test_anonymize_multiple_same_entities(self, anonymizer):
        """Test that the same entity gets the same anonymization"""
        text1 = "My name is John Doe."
        text2 = "John Doe is my name."
        
        anonymizer.anonymize_text(text1)
        anonymized2 = anonymizer.anonymize_text(text2)
        
        # Check the same tag was used in both texts
        assert "<PERSON_1>" in anonymized2

    def test_anonymize_payload_messages(self, anonymizer):
        """Test payload message anonymization"""
        payload = {
            "messages": [
                {"role": "user", "content": "My name is John Doe"},
                {"role": "assistant", "content": "Hello John Doe"}
            ],
            "user": "user123",
            "other_field": "unchanged"
        }
        
        anonymized = anonymizer.anonymize_payload(payload)
        
        # Check messages were anonymized
        assert "John Doe" not in anonymized["messages"][0]["content"]
        assert "John Doe" not in anonymized["messages"][1]["content"]
        assert "<PERSON_" in anonymized["messages"][0]["content"]
        assert "<PERSON_" in anonymized["messages"][1]["content"]
        
        # Check user field was anonymized
        assert "user123" not in anonymized["user"]
        assert "<PERSON_" in anonymized["user"]
        
        # Check other fields unchanged
        assert anonymized["other_field"] == "unchanged"

    def test_deanonymize_payload(self, anonymizer):
        """Test payload deanonymization"""
        original_payload = {
            "messages": [
                {"role": "user", "content": "My name is John Doe"},
                {"role": "assistant", "content": "Hello John Doe"}
            ],
            "user": "user123"
        }
        
        # First anonymize
        anonymized = anonymizer.anonymize_payload(original_payload)
        
        # Then deanonymize
        deanonymized = anonymizer.deanonymize_payload(anonymized)
        
        # Check it matches original
        assert deanonymized == original_payload

    def test_empty_text(self, anonymizer):
        """Test handling of empty text"""
        assert anonymizer.anonymize_text("") == ""
        assert anonymizer.deanonymize_text("") == ""

    def test_no_pii_text(self, anonymizer):
        """Test text with no PII"""
        text = "This is just a normal sentence."
        assert anonymizer.anonymize_text(text) == text
        assert anonymizer.deanonymize_text(text) == text

    def test_multiple_entity_types(self, anonymizer):
        """Test handling of multiple entity types"""
        text = "John Doe lives in New York and works at Google."
        anonymized = anonymizer.anonymize_text(text)
        
        assert "<PERSON_" in anonymized
        assert "<LOCATION_" in anonymized
        assert "<ORG_" in anonymized
        
        deanonymized = anonymizer.deanonymize_text(anonymized)
        assert deanonymized == text

    def test_partial_payload(self, anonymizer):
        """Test payloads with missing fields"""
        payload = {"messages": [{"role": "user"}]}  # No content
        assert anonymizer.anonymize_payload(payload) == payload
        
        payload = {}  # Empty payload
        assert anonymizer.anonymize_payload(payload) == payload

    def test_non_string_fields(self, anonymizer):
        """Test payload with non-string fields"""
        payload = {
            "messages": [{"role": "user", "content": 123}],  # Numeric content
            "user": None,  # None value
            "flag": True  # Boolean value
        }
        
        # Should not raise exceptions
        anonymized = anonymizer.anonymize_payload(payload)
        assert anonymized == payload

    def test_reversible_anonymization(self, anonymizer):
        """Test that anonymization is fully reversible"""
        original = {
            "messages": [
                {"role": "user", "content": "My phone is 555-1234"},
                {"role": "assistant", "content": "I'll call 555-1234"}
            ],
            "user": "test@example.com",
            "session_id": "session123"
        }
        
        anonymized = anonymizer.anonymize_payload(original)
        deanonymized = anonymizer.deanonymize_payload(anonymized)
        
        assert deanonymized == original

    def test_multiple_calls_maintain_state(self, anonymizer):
        """Test that multiple calls maintain proper mapping state"""
        text1 = "Contact Alice at alice@example.com"
        text2 = "Alice's number is 555-1234"
        
        # First call
        anonymized1 = anonymizer.anonymize_text(text1)
        print(anonymizer.analyzer.analyze(text=text1, language="en"))
        assert "<PERSON_" in anonymized1
        assert "<EMAIL_ADDRESS_" in anonymized1
        
        # Second call
        anonymized2 = anonymizer.anonymize_text(text2)
        assert "<PERSON_" in anonymized2
        assert "<PHONE_NUMBER_" in anonymized2
        
        # Check same person tag was used
        person_tag = [word for word in anonymized1.split() if word.startswith("<PERSON_")][0]
        assert person_tag in anonymized2
        
        # Check deanonymization works
        deanonymized1 = anonymizer.deanonymize_text(anonymized1)
        deanonymized2 = anonymizer.deanonymize_text(anonymized2)
        
        assert deanonymized1 == text1
        assert deanonymized2 == text2