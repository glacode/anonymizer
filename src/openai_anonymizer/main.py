from fastapi import FastAPI, HTTPException
import httpx
from anonymizer import OpenAIPayloadAnonymizer
from config import settings
from schemas import OpenAIRequest
import logging

app = FastAPI(title="OpenAI API Anonymizer")
logger = logging.getLogger(__name__)

@app.post("/v1/chat/completions")
async def proxy_openai(request: OpenAIRequest):
    try:
        # Use same instance for both anonymize + deanonymize
        anonymizer = OpenAIPayloadAnonymizer()
        
        # Convert Pydantic model to dict for processing
        payload = request.model_dump(exclude_unset=True)
        
        # Anonymize input
        anonymized_payload = anonymizer.anonymize_payload(payload)
        logger.debug(f"Anonymized payload: {anonymized_payload}")
        
        # Send anonymized request to OpenAI-compatible API
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json"
            }
            response = await client.post(
                settings.openai_api_url,
                json=anonymized_payload,
                headers=headers,
                timeout=30.0
            )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail="Error from OpenAI API"
            )
        
        # Deanonymize output
        openai_response = response.json()
        deanonymized_response = anonymizer.deanonymize_payload(openai_response)
        logger.debug(f"Deanonymized response: {deanonymized_response}")
        
        return deanonymized_response

    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True
    )
