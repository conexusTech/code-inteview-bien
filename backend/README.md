# AI Prompting Project with LangChain x HuggingFace

This project demonstrates of AI-Prompting backend application using LangChain, HuggingFace and FastAPI.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Running Locally](#running-locally)
- [Usage](#usage)

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/backend-AI-project.git
   cd backend-AI-project
2. **Create a virtual environment:** 

    ```
    python3 -m venv venv
3. **Activate the virtual environment:**  

- On macOS/Linux

    ```
    source venv/bin/activate 
- On Windows  

    ```
    venv\Scripts\activate
4. **Install the required libraries in your project:** 
    ```
    pip install -r hf_requirements.txt
## Setup
1. **Set up your OpenAI OR Hugging Face API key:** 
    Create a .env file in the project root and add your API key:
    
    ```
    #for openAi
    OPENAI_API_KEY=api_key

    #for hugging face
    HUGGINGFACE_API_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    HUGGINGFACE_API_TOKEN="api_token"
2. **Export the API to virutual env**
    ```
    #for openai
    export OPENAI_API_KEY=api_key

    #for hugging face
    export HUGGINGFACE_API_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    export HUGGINGFACE_API_TOKEN="your_generated_token"
## Running Locally:

1. Run the application by typing
    ```
    uvicorn main:app --reload #for hugging face
    uvicorn chatopenai:app --reload #fore openai
- **Access the API:**
  - Open your browser and navigate to http://127.0.0.1:8000/docs to view the interactive API documentation.
Test the endpoint using curl
## Usage
Use the endpoint to send questions and receive responses from the AI model.
- /query/ 
    - Params 
        ```
        {
            "question": "Hey",
            "conversation_id": "ceb956ce-1772-411a-9aaa-15adf0cb0676"
        }
