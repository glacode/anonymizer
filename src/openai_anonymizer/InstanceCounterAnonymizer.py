from typing import Dict
from presidio_anonymizer.operators import Operator, OperatorType

# see this example here https://microsoft.github.io/presidio/samples/python/pseudonymization/

class InstanceCounterAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with an instance counter per entity type.
    """

    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: Dict = None) -> str:
        """Anonymize the input text."""

        entity_type: str = params["entity_type"]
        entity_mapping: Dict[str, Dict[str, str]] = params["entity_mapping"]

        # Optional: pass ordered list of entities per type
        ordered_values: list[str] = params.get("ordered_values", {}).get(entity_type, [])

        entity_mapping_for_type = entity_mapping.setdefault(entity_type, {})

        if text in entity_mapping_for_type:
            return entity_mapping_for_type[text]

        # Find its index in the expected order (if possible)
        if ordered_values and text in ordered_values:
            index = ordered_values.index(text)
        else:
            # fallback: assign a new index based on the current size of the mapping
            index = len(entity_mapping_for_type)

        new_text = self.REPLACING_FORMAT.format(entity_type=entity_type, index=index)

        entity_mapping_for_type[text] = new_text
        return new_text

    def validate(self, params: Dict = None) -> None:
        """Validate operator parameters."""

        if not params or "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return "entity_counter"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize
