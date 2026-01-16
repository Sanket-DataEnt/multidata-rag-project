"""AWS Lambda handler for FastAPI application."""
import os
from mangum import Mangum

# Set root path for API Gateway
# This tells FastAPI it's being served under /prod so OpenAPI JSON URLs are correct
os.environ["ROOT_PATH"] = "/prod"

# Create /tmp directories
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/cached_chunks", exist_ok=True)

# Import app after directory creation
from app.main import app

# Initialize services before creating handler (since lifespan="off")
# This runs once when Lambda container starts
from app.main import initialize_services
initialize_services()

# Lambda handler
# API Gateway HTTP API (v2 payload format)
handler = Mangum(app, lifespan="off", api_gateway_base_path="/prod")
