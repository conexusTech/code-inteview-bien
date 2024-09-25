# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r hf_requirements.txt

# Run the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 #change the port as you wish  maybe the host also
