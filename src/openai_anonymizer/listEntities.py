import spacy

# Load your model
nlp = spacy.load("en_core_web_lg")  # or "en_core_web_lg"

# Test with a sample sentence
doc = nlp("John Doe works at Microsoft in New York on January 1st, 2024. It's username is user123 and it password is 123456 and email is john.doe@example.com and phone number is +1234567890.")

# Print detected entities
for ent in doc.ents:
    print(f"Text: {ent.text}, Label: {ent.label_}")

# Expected output for "John Doe works at Microsoft in New York on January 1st, 2024. It's username is user123 and it password is 123456"
# Text: John Doe, Label: PERSON
# Text: Microsoft, Label: ORG
# Text: New York, Label: GPE
# Text: January 1st, 2024, Label: DATE
# Text: 123456, Label: DATE


# https://spacy.io/models/en#en_core_web_sm
# ner 
# CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART