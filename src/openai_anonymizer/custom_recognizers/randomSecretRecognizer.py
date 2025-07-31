# pyright: reportUntypedBaseClass=false

import re
from presidio_analyzer import RecognizerResult, EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts

class RandomSecretRecognizer(EntityRecognizer):
    def __init__(self):
        super().__init__(supported_entities=["RANDOM_SECRET"], name="RandomSecretRecognizer")
        self.MIN_LENGTH = 8

    def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []

        for match in self.find_potential_secrets(text):
            start, end = match.span()
            score = self.estimate_confidence(match.group())
            if score > 0.0:
                results.append(
                    RecognizerResult(
                        entity_type="RANDOM_SECRET",
                        start=start,
                        end=end,
                        score=score
                    )
                )

        return results

    def find_potential_secrets(self, text: str):
        # Basic candidate: long-ish word-like strings (>= MIN_LENGTH)
        pattern = rf"\b[\w!@#$%^&*()\-_=+]{{{self.MIN_LENGTH},}}\b"
        return re.finditer(pattern, text)

    def estimate_confidence(self, value: str) -> float:
        classes = {
            'lower': any(c.islower() for c in value),
            'upper': any(c.isupper() for c in value),
            'digit': any(c.isdigit() for c in value),
            'symbol': any(c in "!@#$%^&*()-_=+" for c in value)
        }
        class_count = sum(classes.values())

        if class_count >= 3:
            return 0.99
        elif class_count >= 2 and (classes['digit'] or classes['symbol']):
            return 0.85
        else:
            return 0.0  # Do not return low-confidence matches
