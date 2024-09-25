import os
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict
from starlette.middleware.base import BaseHTTPMiddleware
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from template import get_template
from fastapi.responses import JSONResponse


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
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Request body format with validation
class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to be answered")
    conversation_id: str = Field(..., min_length=1, max_length=256, description="Unique identifier for each user/chat room")

# Fetching the API and checking if it has been set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Either forgot to execute 'export OPENAI_API_KEY=api_key' or .env OPENAI_API_KEY=api_key does not exist!")

# Set up the LLM Configuration with ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4", 
    api_key=OPENAI_API_KEY
)

# Load the template from template.py
template = get_template()
prompt = PromptTemplate(input_variables=["question"], template=template)

# Dictionary to store conversation histories for each user
conversation_histories: Dict[str, ConversationBufferMemory] = {}

# Function to get or create conversation memory for each user
def get_conversation_memory(conversation_id: str) -> ConversationBufferMemory: 
    if conversation_id not in conversation_histories: 
        conversation_histories[conversation_id] = ConversationBufferMemory()
    return conversation_histories[conversation_id]

# Define the API endpoint to handle queries
@app.post("/query/")
async def get_response(query: Query):
    start_time = time.time()
    try:
        # Get conversation memory for this user/chat room
        memory = get_conversation_memory(query.conversation_id)

        # Create an LLMChain with the prompt and memory
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory
        )

        # Generate a response with conversation history included
        response_content = await chain.arun(question=query.question)  # Asynchronous call to prevent blocking

        # Measure processing time
        process_time = time.time() - start_time
        logging.info(f"Processed request in {process_time:.2f} seconds")

        return JSONResponse(content={"response": response_content})

    except RateLimitExceeded:
        logging.error("Rate limit exceeded")
        return JSONResponse(content={"error": "Rate limit exceeded"}, status_code=429)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": "An internal error occurred. Please try again later."}, status_code=500)

