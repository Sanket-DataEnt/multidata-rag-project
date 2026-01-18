# Multi-Source RAG + Text-to-SQL System

A production-ready FastAPI application that combines **Document RAG (Retrieval-Augmented Generation)** with **Text-to-SQL** capabilities, featuring intelligent query routing, evaluation metrics, and monitoring.

## 🌟 Features

### Core Capabilities
- **📄 Document RAG**: Upload and query documents (PDF, DOCX, CSV, JSON, TXT) using AI-powered retrieval
- **🗄️ Text-to-SQL**: Convert natural language questions to SQL queries with approval workflow
- **🧭 Intelligent Query Routing**: Automatically routes queries to SQL, Documents, or both (HYBRID)

### Advanced Document Processing
- **🔬 Docling Integration**: Context-aware document parsing with HybridChunker
- **📊 Structure Preservation**: Maintains document hierarchy (headings, paragraphs, lists)
- **🎯 Better RAG Quality**: Improved retrieval and RAGAS scores with structure-aware chunks

### Performance & Optimization
- **⚡ Smart Caching**: SHA-256 content-based deduplication for chunks and embeddings
- **💰 Cost Reduction**: 40-60% reduction in OpenAI API calls via cache hits
- **🏗️ ARM64 Architecture**: 20% cheaper Lambda costs compared to x86_64

### Production Deployment
- **☁️ AWS Lambda**: Serverless deployment with automatic scaling
- **🔄 CI/CD Pipeline**: GitHub Actions automatic deployment on push to main
- **📊 CloudWatch Monitoring**: Real-time logs and metrics
- **🌍 Cross-Platform**: Build Docker images on Windows, Mac (Intel/ARM), or Linux

### Evaluation & Monitoring
- **📈 RAGAS Metrics**: Faithfulness and answer relevancy evaluation
- **🔍 OPIK Tracking**: End-to-end request monitoring
- **✅ Input Validation**: Comprehensive validation for file uploads and queries
- **🛡️ Error Handling**: Structured error responses with detailed messages

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Query Routing](#query-routing)
- [Performance Features](#performance-features)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Deployment](#deployment)
  - [Production (AWS Lambda)](#production-aws-lambda-recommended)
  - [Local Development (Docker)](#local-development-docker)
  - [CI/CD Pipeline](#cicd-pipeline-github-actions)
- [Deployment Documentation](#deployment-documentation)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## 🚀 Quick Start

```bash
# 1. Clone the repository
cd multidata-rag-project

# 2. Create virtual environment (Python 3.12+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# OR using UV (faster):
uv pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys (see Configuration section)

# 5. Run the application
uvicorn app.main:app --reload

# 6. Visit the API docs
open http://localhost:8000/docs
```

### Quick Start (AWS Lambda - Production) ⭐

**For production serverless deployment**, see the complete guide:
- 📖 [Complete Deployment Guide](deploy_docs/DEPLOYMENT.md) - Step-by-step Lambda setup
- 🚀 Automatic CI/CD via GitHub Actions - Push to deploy
- ⚡ ARM64 architecture - 20% cost savings vs x86_64
- 📊 CloudWatch monitoring - Real-time logs and metrics

**Your API Endpoint**: `https://{api-id}.execute-api.us-east-1.amazonaws.com/prod`

**Estimated Setup Time**: 30-45 minutes (one-time configuration)

## 📦 Prerequisites

### For Local Development

- **Python 3.12+**
- **OpenAI API Key** (for embeddings and LLM)
- **Pinecone Account** (for vector storage)
  - Create an index with dimension=1536, metric=cosine
- **PostgreSQL Database** (for Text-to-SQL)
  - Supabase recommended for easy setup
- **OPIK API Key** (optional, for monitoring)

### For AWS Lambda Deployment

**All of the above, plus:**
- **AWS Account** with admin access or permissions for ECR, Lambda, IAM, API Gateway
- **AWS CLI** (version 2.x) configured with credentials
- **Docker** for building Lambda container images
- **GitHub Repository** for CI/CD pipeline
- **Estimated Setup Time**: 30-45 minutes (one-time)

📖 **See [Deployment Guide](deploy_docs/DEPLOYMENT.md) for detailed AWS setup instructions**

## 🔧 Installation

### Choose Your Deployment Method

| Deployment | Best For | Setup Time | Infrastructure |
|-----------|----------|------------|----------------|
| **AWS Lambda** ⭐ | Production, variable load, team collaboration | 30-45 min | Serverless (managed by AWS) |
| **Docker** | Local development, self-hosted, testing | 5 minutes | Container (self-managed) |

**Recommendation**: Use **AWS Lambda** for production deployments and **Docker** for local development.

### 1. System Dependencies

**macOS:**
```bash
brew install libmagic poppler tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libmagic1 poppler-utils tesseract-ocr
```

**Windows:**
- Download and install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases)
- Download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Python Packages

```bash
# Using pip
pip install -r requirements.txt

# Using UV (faster, recommended)
uv pip install -r requirements.txt
```

### 3. External Services Setup

#### Pinecone (Vector Database)

1. Sign up at [pinecone.io](https://www.pinecone.io)
2. Create a new index:
   - **Dimensions**: 1536 (for OpenAI text-embedding-3-small)
   - **Metric**: cosine
   - **Region**: us-east-1-aws (or your preferred region)
3. Get your API key from the dashboard

#### Supabase (PostgreSQL Database)

1. Sign up at [supabase.com](https://supabase.com)
2. Create a new project
3. Run the schema from `data/sql/schema.sql` in the SQL editor
4. Optionally, generate sample data:
   ```bash
   python data/generate_sample_data.py
   ```
5. Get your connection string from Project Settings → Database

#### OPIK (Monitoring - Optional)

1. Sign up at [opik.ai](https://www.opik.ai) or run locally
2. Get your API key (optional, works without key in local mode)

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Pinecone Configuration
PINECONE_API_KEY=pcsk_...
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=rag-documents

# Supabase/PostgreSQL Configuration
DATABASE_URL=postgresql://user:password@host:port/database

# OPIK Monitoring (Optional)
OPIK_API_KEY=  # Leave empty for local tracking

# Text Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## 📖 Usage

### 1. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "file_size": "2.5 MB",
  "chunks_created": 15,
  "total_tokens": 7680,
  "message": "Document processed and 15 chunks stored in Pinecone"
}
```

### 3. Query Documents

```bash
curl -X POST "http://localhost:8000/query/documents" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?", "top_k": 3}'
```

### 4. Generate SQL

```bash
curl -X POST "http://localhost:8000/query/sql/generate" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers do we have?"}'
```

### 5. Unified Query (Recommended)

```bash
# Automatically routes to the appropriate service
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Show total revenue and explain our pricing strategy"}'
```

## 🔌 API Endpoints

### Health & Info

- **GET `/health`** - Health check
- **GET `/info`** - System information and available features
- **GET `/`** - Welcome message with quick links

### Document Operations

- **POST `/upload`** - Upload and process documents
  - Supported formats: PDF, DOCX, DOC, CSV, JSON, TXT
  - Max size: 50 MB
  - Returns: chunks created, token count

- **GET `/documents`** - List all uploaded documents
  - Returns: filename, size, upload timestamp

- **POST `/query/documents`** - Query documents using RAG
  - Parameters: `question` (string), `top_k` (int, default=3)
  - Returns: answer, sources, chunks used

### SQL Operations

- **POST `/query/sql/generate`** - Generate SQL from natural language
  - Parameters: `question` (string)
  - Returns: `query_id`, SQL, explanation

- **POST `/query/sql/execute`** - Execute approved SQL query
  - Parameters: `query_id` (string), `approved` (bool)
  - Returns: results, row count

- **GET `/query/sql/pending`** - List pending SQL queries
  - Returns: all queries awaiting approval

### Unified Query (Recommended)

- **POST `/query`** - Intelligent query routing
  - Parameters:
    - `question` (string, required)
    - `auto_approve_sql` (bool, default=false, testing only)
    - `top_k` (int, default=3)
  - Returns: routed response with explanation

## 🧭 Query Routing

The system automatically routes queries based on keyword analysis:

### SQL Queries
Routed to Text-to-SQL service for data retrieval:

**Keywords**: count, total, sum, average, revenue, sales, orders, customers, list all, show all, how many, top, bottom, last, recent, etc.

**Examples:**
- "How many customers do we have?"
- "What is the total revenue from delivered orders?"
- "Show me the top 10 customers by spending"

### Document Queries
Routed to RAG service for information retrieval:

**Keywords**: what is, explain, define, policy, procedure, guide, manual, how to, why, according to, etc.

**Examples:**
- "What is our return policy?"
- "Explain the customer complaint procedure"
- "How should I process a refund?"

### Hybrid Queries
Routed to both services, combining data with context:

**Keywords**: and explain, and describe, show data and explain, etc.

**Examples:**
- "Show total revenue by segment and explain our segmentation strategy"
- "List top products and describe pricing policies"

## ⚡ Performance Features

### Smart Caching System

The application includes an intelligent caching layer that significantly reduces API costs and improves response times.

**How It Works:**
- **Content-Based Hashing**: Uses SHA-256 to generate unique document IDs from file contents (not filenames)
- **Chunk & Embedding Cache**: Stores processed chunks and their embeddings locally
- **Automatic Deduplication**: Re-uploading the same document (even with different filename) hits cache

**Benefits:**
- **40-60% Cost Reduction**: Dramatically reduces OpenAI API calls for embeddings
- **Faster Processing**: Cached documents process in milliseconds vs seconds
- **Smart Invalidation**: Changed content automatically gets new hash and fresh processing

**Example:**
```python
# First upload - Full processing
POST /upload (document.pdf) → 15 chunks created, 15 embeddings generated
Cost: $0.0003

# Re-upload same content (even as "copy.pdf") - Cache hit
POST /upload (copy.pdf) → 15 chunks loaded from cache
Cost: $0.0000 ✅ 100% savings
```

**Cache Locations:**
- **Local Development**: `data/cached_chunks/`
- **AWS Lambda**: `/tmp/cached_chunks/` (ephemeral, survives ~15 minutes)

**Implementation**: See `app/services/cache_service.py` for details

---

### Advanced Document Processing (Docling)

The application uses **Docling** for context-aware document parsing with structure preservation.

**What is Docling?**
- Advanced document understanding library from IBM Research
- Preserves document structure (headings, paragraphs, lists, tables)
- Context-aware chunking via HybridChunker

**Key Features:**
- **Structure-Aware Chunking**: Chunks respect document hierarchy
- **Heading Context**: Each chunk includes its parent heading for better retrieval
- **Better RAG Quality**: Improved answer accuracy and RAGAS scores
- **Automatic Fallback**: Falls back to Unstructured.io if Docling unavailable

**Processing Pipeline:**
```
Document Upload
  ↓
Docling Parser (primary)
  ↓
HybridChunker (structure-aware)
  ↓
Chunks with heading context
  ↓
Embeddings + Vector Storage
```

**Supported Formats:**
- PDF, DOCX, DOC (full layout analysis)
- CSV, JSON, TXT (structured parsing)

**Implementation**: See `app/services/docling_service.py` for details

---

### ARM64 Lambda Architecture

The Lambda deployment uses **ARM64 (Graviton2)** architecture for optimal cost and performance.

**Benefits:**
- **20% Cost Savings**: ARM64 Lambda is 20% cheaper than x86_64
- **Better Price/Performance**: More efficient processing per dollar
- **Cross-Platform Builds**: Build on any platform (Windows, Mac Intel/ARM, Linux)

**Technical Specs:**
- **Platform**: `linux/arm64`
- **Memory**: 8192 MB (8 GB)
- **Timeout**: 900 seconds (15 minutes)
- **Storage**: 10240 MB (10 GB /tmp directory)

**Cost Comparison** (100K requests/month, 30s avg duration, 8GB RAM):

| Architecture | Monthly Cost | Savings |
|--------------|--------------|---------|
| x86_64 | ~$50-81 | - |
| **ARM64** ⭐ | **~$40-65** | **~$10-16/month (20%)** |

**Build Command:**
```bash
# Cross-platform build for ARM64 Lambda
docker build --platform linux/arm64 -f Dockerfile.lambda -t my-lambda:latest .
```

📖 **See [Cross-Platform Build Guide](deploy_docs/CROSS_PLATFORM_BUILD.md) for detailed instructions**

## 📊 Evaluation

Run the RAGAS evaluation to measure system quality:

```bash
python evaluate.py
```

**Metrics:**
- **Faithfulness** (target > 0.7): Answer accuracy based on retrieved context
- **Answer Relevancy** (target > 0.8): How well the answer matches the question

**Output:**
- Console: Real-time progress and scores
- File: `evaluation_results.json` with detailed results

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       v
┌──────────────────────────────────────────────────┐
│         Deployment Options                       │
│  Local/Docker  ←→  AWS Lambda + API Gateway     │
└──────┬───────────────────────────────────────────┘
       │
       v
┌──────────────────────────────────────────────────┐
│       FastAPI Application                        │
│  (OPIK Monitoring + CloudWatch Logging)         │
└──────┬───────────────────────────────────────────┘
       │
       v
┌──────────────────┐
│  Query Router    │ ← Keyword-based routing
└──────┬───────────┘
       │
       ├─────────────┬─────────────────┬──────────────┐
       │             │                 │              │
       v             v                 v              v
   [SQL Path]   [Documents]       [HYBRID]      [Cache Check]
       │             │                 │              │
       v             v                 │              v
┌──────────┐  ┌──────────┐           │         ┌────────────┐
│  Vanna   │  │ Docling  │           │         │ SHA-256    │
│ SQL Gen  │  │   +      │           │         │ Content    │
│          │  │   RAG    │           │         │ Hash       │
└────┬─────┘  └────┬─────┘           │         └────┬───────┘
     │             │                  │              │
     v             v                  v              v
┌──────────┐  ┌──────────┐     ┌──────────┐   ┌──────────┐
│PostgreSQL│  │ Pinecone │     │   Both   │   │  Cache   │
└──────────┘  └──────────┘     └──────────┘   │ (chunks/ │
                                               │embeddings)
                                               └──────────┘
```

### Components

**Core Services:**
- **Document Service**: Parses PDF/DOCX/CSV/JSON using Docling (primary) + Unstructured.io (fallback)
- **Docling Service**: Context-aware parsing with HybridChunker for structure preservation
- **Embedding Service**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector Service**: Pinecone with gRPC for fast vector operations
- **RAG Service**: Retrieval + GPT-4 generation with source citations
- **SQL Service**: Vanna.ai for Text-to-SQL with training on schema
- **Cache Service**: SHA-256 content-based deduplication for chunks and embeddings

**Routing & Validation:**
- **Query Router**: Keyword-based intelligent routing (SQL/Documents/Hybrid)
- **Validation**: File type/size, query length, SQL safety checks

**Deployment Options:**
- **AWS Lambda (Production)**: ARM64 serverless with API Gateway, CloudWatch, CI/CD
- **Docker (Development)**: Local containerized deployment for testing

**Monitoring:**
- **OPIK Tracking**: End-to-end request monitoring on all key endpoints
- **CloudWatch Logs**: Real-time Lambda logs and metrics (production only)


## 👨‍💻 Development

### Project Structure

```
multidata-rag-project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with endpoints
│   ├── config.py            # Pydantic settings
│   ├── utils.py             # Validation and error handling
│   └── services/
│       ├── document_service.py   # Document parsing & chunking
│       ├── embedding_service.py  # OpenAI embeddings
│       ├── vector_service.py     # Pinecone operations
│       ├── rag_service.py        # RAG pipeline
│       ├── sql_service.py        # Vanna Text-to-SQL
│       └── router_service.py     # Query routing
├── data/
│   ├── uploads/             # Uploaded documents (gitignored)
│   ├── sql/
│   │   └── schema.sql       # Database schema
│   └── generate_sample_data.py  # Sample data generator
├── tests/
│   └── test_queries.json    # Evaluation test queries
├── evaluate.py              # RAGAS evaluation script
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

### Running Tests

```bash
# Run evaluation
python evaluate.py

# Test individual endpoints
curl http://localhost:8000/health
curl http://localhost:8000/info
```

### Code Style

- **Type hints**: All functions have type annotations
- **Docstrings**: Google-style docstrings for all public functions
- **Validation**: Input validation on all endpoints
- **Error handling**: Structured error responses


## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For issues and questions:

- Create an issue in the GitHub repository
- Check the Troubleshooting section above
- Review the API documentation at `/docs`

## 🎯 Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Document Upload | All formats work | Test with PDF, DOCX, CSV, JSON |
| Document Retrieval | Top-3 relevant chunks | Manual review of query results |
| SQL Generation | 70%+ accuracy | Run evaluate.py |
| Query Routing | 80%+ correct | Test with mixed queries |
| RAGAS Faithfulness | > 0.7 | Run evaluate.py |
| RAGAS Relevancy | > 0.8 | Run evaluate.py |
| Response Time | < 15 seconds | Monitor OPIK dashboard |

## 🚀 Deployment

### Deployment Overview

| Feature | AWS Lambda ⭐ | Docker (Local) |
|---------|--------------|----------------|
| **Best For** | Production, team collaboration | Local development, testing |
| **Scaling** | Automatic (0 to 1000s) | Manual (single container) |
| **Cost** | Pay per use (~$127-227/month) | Self-hosted (compute costs) |
| **Deployment** | Git push (CI/CD) | Docker build + run |
| **Monitoring** | CloudWatch + OPIK | OPIK + file logs |
| **Setup Time** | 30-45 min (one-time) | 5 minutes |

---

### Production (AWS Lambda) ⭐ RECOMMENDED

Deploy your application as a serverless Lambda function with automatic scaling and CI/CD.

#### What You Get

- ☁️ **Serverless**: No servers to manage, automatic scaling
- 🔄 **CI/CD**: Push to `main` → Automatic deployment via GitHub Actions
- ⚡ **ARM64 Optimized**: 20% cheaper than x86_64 Lambda costs
- 📊 **CloudWatch Monitoring**: Real-time logs and metrics
- 🌐 **API Gateway**: HTTPS endpoint with /prod base path
- 💾 **Ephemeral Storage**: 10GB /tmp directory for document processing

#### Technical Specs

- **Platform**: linux/arm64 (Graviton2)
- **Memory**: 8192 MB (8 GB)
- **Timeout**: 900 seconds (15 minutes)
- **Storage**: 10240 MB (10 GB /tmp)
- **Entry Point**: `lambda_handler.handler` (Mangum adapter)

#### Quick Deploy Guide

**Step 1: Initial Setup** (one-time, 30-45 minutes)


This creates:
- Lambda function with ARM64 architecture
- ECR container registry for Docker images
- API Gateway HTTP API with /prod stage
- IAM roles and permissions
- GitHub Actions secrets for CI/CD

**Step 2: Deploy Your Code** (automatic, ~15 minutes)

```bash
# Make your changes
git add .
git commit -m "Update feature"

# Push to main - GitHub Actions handles the rest!
git push origin main
```

GitHub Actions automatically:
1. ✓ Builds ARM64 Docker image
2. ✓ Pushes to Amazon ECR
3. ✓ Updates Lambda function
4. ✓ Runs health check tests
5. ✓ Notifies you of success/failure

**Step 3: Your API is Live!**

```
Base URL: https://{api-id}.execute-api.us-east-1.amazonaws.com

Endpoints:
  - GET  /prod/health          - Health check
  - GET  /prod/info            - System information
  - GET  /prod/docs            - Swagger UI
  - POST /prod/upload          - Upload documents
  - POST /prod/query           - Unified query endpoint
  - POST /prod/query/documents - Document RAG
  - POST /prod/query/sql/*     - Text-to-SQL
```

#### Key Files

**Lambda-Specific:**
- `Dockerfile.lambda` - Optimized Lambda image (no OCR)
- `Dockerfile.lambda.with-tesseract` - Full image with Tesseract OCR
- `lambda_handler.py` - Lambda entry point with Mangum adapter

**CI/CD:**
- `.github/workflows/deploy.yml` - Automatic deployment pipeline
- `.github/workflows/test.yml` - PR testing workflow

**Documentation:**
- `deploy_docs/DEPLOYMENT.md` - Complete deployment guide
- `deploy_docs/DEPLOYMENT_FIXES.md` - Troubleshooting guide
- `deploy_docs/CROSS_PLATFORM_BUILD.md` - Build on any platform
- `deploy_docs/TEAM_SETUP.md` - Team member onboarding

#### Cost Estimate

**AWS Services** (100K requests/month, 30s avg, 8GB RAM, ARM64):

| Service | Cost/Month |
|---------|------------|
| Lambda (ARM64) | $40-65 |
| API Gateway | $1.00 |
| ECR | $0.50 |
| CloudWatch Logs | $5.00 |
| Data Transfer | $0.90 |
| **Total (AWS)** | **~$47-72** |

**External Services**:

| Service | Cost/Month |
|---------|------------|
| OpenAI | $10-30 |
| Pinecone | $70-100 |
| Supabase | $0-25 |
| **Total (External)** | **~$80-155** |

**Grand Total**: **~$127-227/month**

💡 **ARM64 saves ~$10-16/month (20%) compared to x86_64**

---

### Local Development (Docker)

Use Docker for local development and testing.

#### Quick Start with Docker Compose (Recommended)

```bash
# 1. Ensure .env file is configured
cp .env.example .env
# Edit .env with your API keys

# 2. Build and start the container
docker-compose up -d

# 3. View logs
docker-compose logs -f

# 4. API is available at http://localhost:8000

# 5. Stop the container
docker-compose down
```

#### Manual Docker Build

```bash
# Build the image
docker build -t rag-text-to-sql:latest .

# Run the container
docker run -d \
  --name rag-text-to-sql \
  -p 8000:8000 \
  -v $(pwd)/data/uploads:/app/data/uploads \
  -v $(pwd)/data/vanna_chromadb:/app/data/vanna_chromadb \
  --env-file .env \
  rag-text-to-sql:latest

# View logs
docker logs -f rag-text-to-sql

# Stop and remove
docker stop rag-text-to-sql
docker rm rag-text-to-sql
```

#### Docker Features

- **Multi-stage build**: Optimized image size (~800 MB)
- **Health checks**: Automatic health monitoring
- **Persistent volumes**: Documents and training data preserved
- **System dependencies**: All required packages pre-installed
- **Production-ready**: Runs with uvicorn, proper signal handling

#### When to Use Docker vs Lambda

**Use Docker for:**
- ✅ Local development and testing
- ✅ Self-hosted deployment on your own servers
- ✅ Learning the application without AWS setup
- ✅ Environments with strict data residency requirements

**Use Lambda for:**
- ⭐ Production deployments
- ⭐ Team collaboration with CI/CD
- ⭐ Variable load with automatic scaling
- ⭐ Minimal infrastructure management

---

### CI/CD Pipeline (GitHub Actions)

Automatic deployment on every push to `main` branch.

#### Workflow File

`.github/workflows/deploy.yml` - Automatic deployment pipeline

#### Trigger

```bash
git push origin main  # Automatically triggers deployment
```

#### Pipeline Steps

1. **Checkout Code** - Fetches latest code from repository
2. **Setup QEMU** - Enables ARM64 emulation for cross-platform builds
3. **Setup Buildx** - Configures Docker multi-platform builds
4. **Configure AWS** - Authenticates with AWS using GitHub secrets
5. **Login to ECR** - Authenticates with Amazon container registry
6. **Build Image** - Builds ARM64 Docker image (10-15 minutes)
7. **Push to ECR** - Pushes image with multiple tags (latest, arm64, SHA)
8. **Update Lambda** - Updates Lambda function with new image
9. **Wait for Update** - Waits for Lambda to finish updating (30-60s)
10. **Test Deployment** - Runs health check and smoke tests

#### Duration

**Total Time**: ~15-20 minutes per deployment

#### Required GitHub Secrets

Add these in **Settings → Secrets → Actions**:
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_ACCOUNT_ID` - 12-digit AWS account ID
- `API_GATEWAY_URL` - Your API Gateway endpoint URL

#### Viewing Deployment Logs

1. Go to **Actions** tab in GitHub
2. Click on the workflow run
3. Expand steps to see detailed logs

#### Deployment Status

- ✅ **Green checkmark** - Deployment successful
- ❌ **Red X** - Deployment failed (check logs for errors)
- 🟡 **Yellow dot** - Deployment in progress

---

### Monitoring & Logs

#### AWS Lambda (CloudWatch)

**View real-time logs:**
```bash
aws logs tail /aws/lambda/rag-text-to-sql --follow
```

**View logs from last hour:**
```bash
aws logs tail /aws/lambda/rag-text-to-sql --since 1h
```

**Search for errors:**
```bash
aws logs tail /aws/lambda/rag-text-to-sql --filter-pattern "ERROR"
```

**CloudWatch Console:**
- Go to AWS Console → CloudWatch → Log groups
- Find `/aws/lambda/rag-text-to-sql`
- View logs, create dashboards, set alarms

#### Docker (Local)

**Container logs:**
```bash
docker logs -f rag-text-to-sql
```

**Application log files:**
```bash
# Inside container
docker exec rag-text-to-sql cat /app/logs/app.log
```

---

### Updating Your Deployment

#### Lambda (Automatic via CI/CD)

```bash
# Make code changes
vim app/main.py

# Commit and push - CI/CD handles deployment
git add .
git commit -m "Add new feature"
git push origin main

# GitHub Actions automatically:
# 1. Builds new Docker image
# 2. Pushes to ECR
# 3. Updates Lambda function
# 4. Runs tests
```

#### Lambda (Manual)

```bash
# Build image for ARM64
docker build --platform linux/arm64 -f Dockerfile.lambda -t $ECR_URI:manual .

# Push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URI
docker push $ECR_URI:manual

# Update Lambda
aws lambda update-function-code \
  --function-name rag-text-to-sql \
  --image-uri $ECR_URI:manual \
  --architectures arm64
```

#### Docker

```bash
# Rebuild and restart
docker-compose build
docker-compose up -d

# Or manually
docker build -t rag-text-to-sql:latest .
docker stop rag-text-to-sql
docker rm rag-text-to-sql
docker run -d --name rag-text-to-sql ... rag-text-to-sql:latest
```


### Quick Reference

**Lambda Architecture Specs:**
- **Platform**: linux/arm64 (Graviton2)
- **Memory**: 8192 MB (8 GB)
- **Timeout**: 900 seconds (15 minutes)
- **Storage**: 10240 MB (10 GB /tmp)
- **Entry Point**: `lambda_handler.handler`

**CI/CD Pipeline:**
- **Trigger**: Push to `main` branch
- **Duration**: ~15-20 minutes
- **Steps**: 10 automated steps (build, push, update, test)
- **Workflow**: `.github/workflows/deploy.yml`

**Monitoring:**
- **CloudWatch Logs**: `/aws/lambda/rag-text-to-sql`
- **Metrics**: Invocations, errors, duration, throttles
- **Alarms**: Configure in CloudWatch console

**Cost Estimate** (100K requests/month):
- AWS Services: ~$47-72/month
- External Services: ~$80-155/month
- **Total**: ~$127-227/month

**API Endpoints:**
```
https://{api-id}.execute-api.us-east-1.amazonaws.com/prod/health
https://{api-id}.execute-api.us-east-1.amazonaws.com/prod/docs
https://{api-id}.execute-api.us-east-1.amazonaws.com/prod/query
```

## 📚 Additional Resources

### Core Technologies

- [FastAPI Documentation](https://fastapi.tiangolo.com) - Modern Python web framework
- [Pinecone Documentation](https://docs.pinecone.io) - Vector database for embeddings
- [Vanna.ai Documentation](https://vanna.ai/docs) - Text-to-SQL generation
- [RAGAS Documentation](https://docs.ragas.io) - RAG evaluation framework
- [OPIK Documentation](https://www.opik.ai/docs) - LLM monitoring and tracking
- [Docling Documentation](https://github.com/DS4SD/docling) - Advanced document understanding
- [OpenAI API Documentation](https://platform.openai.com/docs) - Embeddings and LLM

### AWS Deployment

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/) - Serverless compute
- [API Gateway HTTP API Guide](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api.html) - HTTP API creation
- [Amazon ECR Documentation](https://docs.aws.amazon.com/ecr/) - Container registry
- [CloudWatch Logs Documentation](https://docs.aws.amazon.com/cloudwatch/) - Logging and monitoring
- [GitHub Actions Documentation](https://docs.github.com/en/actions) - CI/CD workflows
- [Docker Multi-Platform Builds](https://docs.docker.com/build/building/multi-platform/) - ARM64 cross-compilation

### Project Deployment Guides

**Start here for deployment**:
- 📖 [Complete Deployment Guide](deploy_docs/DEPLOYMENT.md) - Step-by-step Lambda setup
- 🔧 [Deployment Troubleshooting](deploy_docs/DEPLOYMENT_FIXES.md) - Fix common issues
- 👥 [Team Setup Guide](deploy_docs/TEAM_SETUP.md) - Onboard new team members
- 🌍 [Cross-Platform Builds](deploy_docs/CROSS_PLATFORM_BUILD.md) - Build on any OS

---

**Built with ❤️ using FastAPI, OpenAI, Pinecone, Vanna.ai, Docling, and AWS Lambda**
