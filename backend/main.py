import os
import time
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from template import get_template  # Import the prompt template

# Initialize FastAPI app
app = FastAPI()

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Middleware to handle rate limiting
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
    except RateLimitExceeded as e:
        logging.error(f"Rate limit exceeded: {str(e)}")
        return JSONResponse(content={"error": "Rate limit exceeded"}, status_code=429)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str = Field(..., min_length=1, max_length=256)

# Creating an object array for conversation history
conversation_history = {}

# Fetch the Hugging Face API token from environment variable
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")

huggingface_endpoint_url = os.getenv("HUGGINGFACE_API_URL")
if not huggingface_endpoint_url:
    raise ValueError("HUGGINGFACE_API_URL environment variable not set.")

# Initialize Hugging Face LLM endpoint
llm = HuggingFaceEndpoint(
    endpoint_url=huggingface_endpoint_url,
    temperature=0.1,
    huggingfacehub_api_token=huggingface_token
)

@app.post("/query/")
@limiter.limit("5/minute")
async def get_response(query: Query, request: Request):
    start_time = time.time()

    # Retrieve past conversation based on conversation_id
    past_conversation = conversation_history.get(query.conversation_id, [])
    
    # Format past conversation to provide context
    formatted_conversation = "\n".join(past_conversation)

    # Construct the prompt using the template
    prompt_template = get_template()
    full_input = prompt_template.format(question=query.question)

    try:
        # Generate AI Response
        response = llm.invoke(full_input)

        # Save the past conversation based on conversation_id
        conversation_history[query.conversation_id] = past_conversation + [f"User: {query.question}", f"AI: {response}"]

        # For logging of process time for performance tracking.
        process_time = time.time() - start_time
        logging.info(f"Processed request in {process_time:.2f} seconds")

        return JSONResponse(content={"response": response})

    except RateLimitExceeded:
        logging.error("Rate limit exceeded")
        return JSONResponse(content={"error": "Rate limit exceeded"}, status_code=429)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": "An internal error occurred. Please try again later."}, status_code=500)
