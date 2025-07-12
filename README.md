# OpenAI API Anonymizer Proxy

A privacy-preserving proxy for OpenAI API that automatically detects and anonymizes Personally Identifiable Information (PII) before forwarding requests to OpenAI.

## Features

- Uses Microsoft Presidio for state-of-the-art PII detection
- Consistent anonymization with salt-based hashing
- Fully compatible with OpenAI API specification
- Easy to deploy as a local proxy

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with your OpenAI API key
4. Run: `python -m src.main`

## Usage

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "My email is user@example.com and phone is 555-123-4567"}
    ]
}'