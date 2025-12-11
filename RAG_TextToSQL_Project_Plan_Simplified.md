# Multi-Source RAG + Text-to-SQL Project Plan (Simplified)

**Project Name:** Multi-Source RAG with Vector Database + Text-to-SQL Support  
**Duration:** 4-5 Weeks  
**Complexity Level:** Intermediate  
**Focus:** Core Features Only - MVP Approach

---

## 🎯 Project Overview

### Goal
Build a working RAG system that:
1. Ingests PDF/DOCX/CSV/JSON documents into a vector database
2. Answers questions from documents OR PostgreSQL database
3. Uses Vanna.ai for natural language to SQL conversion
4. Requires user approval before executing SQL
5. Runs in a single Docker container

### Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| **API Framework** | FastAPI | Simple, auto-documentation |
| **Language** | Python 3.12 | Latest stable version |
| **Database** | Supabase (PostgreSQL) | Managed, easy setup |
| **Vector Store** | ChromaDB | Simple, no server needed |
| **Embeddings** | OpenAI text-embedding-3-small | API-based, no model hosting |
| **LLM** | OpenAI GPT-4 | API-based generation |
| **Text-to-SQL** | Vanna.ai | Handles SQL complexity |
| **Document Parsing** | Unstructured.io | One library for all formats |
| **Chunking** | 512 tokens, 50 overlap | Standard approach |
| **Evaluation** | RAGAS | Simple library |
| **Monitoring** | OPIK | Decorator-based tracking |
| **Deployment** | Docker | Single container |

---

## 📁 Project Structure

```
project/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Configuration management
│   └── services/
│       ├── document_service.py  # Document processing
│       ├── embedding_service.py # OpenAI embeddings
│       ├── vector_service.py    # ChromaDB operations
│       ├── sql_service.py       # Vanna.ai integration
│       ├── rag_service.py       # RAG pipeline
│       └── router_service.py    # Query routing
├── data/
│   ├── uploads/                 # Uploaded documents
│   └── chromadb/                # Vector DB persistence
├── tests/
│   └── test_queries.json        # 10 test questions
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── .env.example                 # Template for .env
├── Dockerfile                   # Single image
├── README.md                    # Project documentation
└── evaluate.py                  # RAGAS evaluation script
```

---

## 📦 Core Dependencies

```txt
# API
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0
python-multipart==0.0.6

# Document Processing
unstructured[all-docs]==0.12.0
python-magic-bin==0.4.14  # For file type detection

# Embeddings & Vector DB
openai==1.10.0
chromadb==0.4.22
tiktoken==0.5.2

# Text-to-SQL
vanna[chromadb]==0.3.4
sqlalchemy==2.0.25
psycopg2-binary==2.9.9

# Evaluation & Monitoring
ragas==0.1.5
datasets==2.16.1
opik==0.1.15

# Utilities
python-dotenv==1.0.0
```

---

## 🗓️ Week-by-Week Implementation Plan

---

## **WEEK 1: Setup + Document Ingestion**

### **Day 1-2: Environment Setup**

#### Tasks:
1. **Create Project Structure**
   - Create all folders as per structure above
   - Initialize Git repository
   - Create .gitignore (exclude .env, data/, __pycache__)

2. **Install Dependencies**
   - Create virtual environment: `python -m venv venv`
   - Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
   - Create requirements.txt
   - Install: `pip install -r requirements.txt`
   - Verify imports work

3. **Setup External Services**
   - **Supabase:**
     - Create project at supabase.com
     - Copy connection string (Settings → Database)
     - Test connection
   
   - **OpenAI:**
     - Get API key from platform.openai.com
     - Test API call
   
   - **OPIK:**
     - Sign up at opik.ai
     - Get API key
     - Create project

4. **Create Configuration**
   - Create `.env` file:
     ```env
     OPENAI_API_KEY=sk-...
     DATABASE_URL=postgresql://...
     OPIK_API_KEY=...
     CHUNK_SIZE=512
     CHUNK_OVERLAP=50
     ```
   
   - Create `app/config.py`:
     ```python
     from pydantic_settings import BaseSettings
     
     class Settings(BaseSettings):
         OPENAI_API_KEY: str
         DATABASE_URL: str
         OPIK_API_KEY: str
         CHUNK_SIZE: int = 512
         CHUNK_OVERLAP: int = 50
         
         class Config:
             env_file = ".env"
     ```

5. **Create Basic FastAPI App**
   - Create `app/main.py` with:
     - `/health` endpoint
     - `/info` endpoint
   - Run: `uvicorn app.main:app --reload`
   - Visit: http://localhost:8000/docs
   - Test endpoints work

#### Deliverables:
- ✅ Working development environment
- ✅ All dependencies installed
- ✅ FastAPI app running with /health endpoint
- ✅ External services configured and tested

---

### **Day 3-5: Document Processing Pipeline**

#### Goal:
Upload any document (PDF/DOCX/CSV/JSON) → Extract text → Chunk → Embed → Store in ChromaDB

#### Implementation Steps:

1. **Document Parser (ONE function for all types)**
   
   **File: `app/services/document_service.py`**
   
   ```python
   from unstructured.partition.auto import partition
   
   def parse_document(file_path: str) -> str:
       """
       Parse any document type and return text
       Uses Unstructured.io - handles PDF, DOCX, CSV, JSON
       """
       elements = partition(filename=file_path)
       text = "\n\n".join([str(el) for el in elements])
       return text
   ```

2. **Text Chunking**
   
   **File: `app/services/document_service.py`**
   
   ```python
   import tiktoken
   
   def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
       """
       Split text into overlapping chunks
       Returns list of chunk dictionaries
       """
       tokenizer = tiktoken.encoding_for_model("gpt-4")
       tokens = tokenizer.encode(text)
       chunks = []
       
       for i in range(0, len(tokens), chunk_size - overlap):
           chunk_tokens = tokens[i:i + chunk_size]
           chunk_text = tokenizer.decode(chunk_tokens)
           chunks.append({
               'text': chunk_text,
               'chunk_index': len(chunks),
               'token_count': len(chunk_tokens)
           })
       
       return chunks
   ```

3. **Embedding Generation**
   
   **File: `app/services/embedding_service.py`**
   
   ```python
   import openai
   
   async def generate_embeddings(texts: list[str]) -> list[list[float]]:
       """
       Generate embeddings using OpenAI API
       Batch process for efficiency
       """
       response = await openai.embeddings.create(
           model="text-embedding-3-small",
           input=texts
       )
       return [item.embedding for item in response.data]
   ```

4. **ChromaDB Storage**
   
   **File: `app/services/vector_service.py`**
   
   ```python
   import chromadb
   
   class VectorStore:
       def __init__(self, persist_directory: str = "./data/chromadb"):
           self.client = chromadb.PersistentClient(path=persist_directory)
           self.collection = self.client.get_or_create_collection(
               name="documents",
               metadata={"hnsw:space": "cosine"}
           )
       
       def add_documents(self, chunks: list, embeddings: list, filename: str):
           """Store chunks with embeddings in ChromaDB"""
           ids = [f"{filename}_{i}" for i in range(len(chunks))]
           documents = [chunk['text'] for chunk in chunks]
           metadatas = [
               {
                   'filename': filename,
                   'chunk_index': chunk['chunk_index'],
                   'token_count': chunk['token_count']
               }
               for chunk in chunks
           ]
           
           self.collection.add(
               ids=ids,
               embeddings=embeddings,
               documents=documents,
               metadatas=metadatas
           )
   ```

5. **Upload API Endpoint**
   
   **File: `app/main.py`**
   
   ```python
   from fastapi import FastAPI, UploadFile, File
   
   @app.post("/upload")
   async def upload_document(file: UploadFile = File(...)):
       """
       Upload and process a document
       Steps: save → parse → chunk → embed → store
       """
       # Save file
       file_path = f"./data/uploads/{file.filename}"
       with open(file_path, "wb") as f:
           f.write(await file.read())
       
       # Process pipeline
       text = parse_document(file_path)
       chunks = chunk_text(text)
       
       # Generate embeddings (batch)
       texts = [chunk['text'] for chunk in chunks]
       embeddings = await generate_embeddings(texts)
       
       # Store in ChromaDB
       vector_store.add_documents(chunks, embeddings, file.filename)
       
       return {
           "filename": file.filename,
           "chunks_created": len(chunks),
           "status": "success"
       }
   ```

#### Testing:
- Upload a PDF via /docs interface
- Check `data/chromadb/` for stored data
- Verify no errors in console

#### Deliverables:
- ✅ Can upload PDF/DOCX/CSV/JSON
- ✅ Documents are parsed successfully
- ✅ Text is chunked (512 tokens, 50 overlap)
- ✅ Embeddings are generated via OpenAI
- ✅ Everything stored in ChromaDB

---

## **WEEK 2: Text-to-SQL with Vanna.ai**

### **Day 6-7: Database Setup**

#### Goal:
Create simple 3-table schema in Supabase with sample data

#### Database Schema:

```sql
-- Table 1: Customers (100 rows)
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    segment VARCHAR(20) CHECK (segment IN ('SMB', 'Enterprise', 'Individual')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Orders (200 rows)
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    order_date DATE NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('Pending', 'Delivered', 'Cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 3: Products (50 rows)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT DEFAULT 0
);

-- Create indexes
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
```

#### Data Generation Script:

**File: `data/generate_sample_data.py`**

```python
from faker import Faker
import random
import psycopg2

fake = Faker()

# Connect to Supabase
conn = psycopg2.connect("YOUR_DATABASE_URL")
cur = conn.cursor()

# Generate 100 customers
for _ in range(100):
    cur.execute("""
        INSERT INTO customers (name, email, segment)
        VALUES (%s, %s, %s)
    """, (
        fake.name(),
        fake.email(),
        random.choice(['SMB', 'Enterprise', 'Individual'])
    ))

# Generate 50 products
categories = ['Electronics', 'Software', 'Hardware', 'Services']
for i in range(50):
    cur.execute("""
        INSERT INTO products (name, category, price, stock_quantity)
        VALUES (%s, %s, %s, %s)
    """, (
        f"Product {i}",
        random.choice(categories),
        round(random.uniform(10, 1000), 2),
        random.randint(0, 500)
    ))

# Generate 200 orders
cur.execute("SELECT id FROM customers")
customer_ids = [row[0] for row in cur.fetchall()]

for _ in range(200):
    cur.execute("""
        INSERT INTO orders (customer_id, order_date, total_amount, status)
        VALUES (%s, %s, %s, %s)
    """, (
        random.choice(customer_ids),
        fake.date_between(start_date='-1y', end_date='today'),
        round(random.uniform(50, 5000), 2),
        random.choice(['Pending', 'Delivered', 'Cancelled'])
    ))

conn.commit()
print("✅ Sample data generated successfully!")
```

#### Deliverables:
- ✅ 3 tables created in Supabase
- ✅ 100 customers, 50 products, 200 orders inserted
- ✅ Foreign keys and indexes working
- ✅ Can query data successfully

---

### **Day 8-10: Vanna.ai Integration**

#### Goal:
Natural language question → SQL → Results

#### Implementation:

**File: `app/services/sql_service.py`**

```python
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
import sqlalchemy

class VannaSQLService(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

class TextToSQLService:
    def __init__(self, database_url: str, openai_api_key: str):
        self.vn = VannaSQLService(config={
            'api_key': openai_api_key,
            'model': 'gpt-4-turbo-preview'
        })
        
        # Connect to database
        self.engine = sqlalchemy.create_engine(database_url)
        self.vn.connect_to_postgres(url=database_url)
    
    def train(self):
        """Train Vanna on your database schema"""
        # Auto-learn schema
        df_ddl = self.vn.run_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        self.vn.train(ddl=df_ddl)
        
        # Add golden examples
        training_data = [
            {
                "question": "How many customers do we have?",
                "sql": "SELECT COUNT(*) FROM customers;"
            },
            {
                "question": "What is the total revenue from all orders?",
                "sql": "SELECT SUM(total_amount) FROM orders;"
            },
            {
                "question": "List all orders with status Delivered",
                "sql": "SELECT * FROM orders WHERE status = 'Delivered';"
            },
            {
                "question": "How many orders per customer?",
                "sql": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;"
            },
            {
                "question": "Average order value by customer segment",
                "sql": """SELECT c.segment, AVG(o.total_amount) 
                         FROM customers c 
                         JOIN orders o ON c.id = o.customer_id 
                         GROUP BY c.segment;"""
            }
        ]
        
        for example in training_data:
            self.vn.train(question=example["question"], sql=example["sql"])
    
    def generate_sql(self, question: str) -> dict:
        """Generate SQL from natural language question"""
        sql = self.vn.generate_sql(question=question)
        
        return {
            "sql": sql,
            "question": question
        }
    
    def execute_sql(self, sql: str):
        """Execute SQL and return results"""
        df = self.vn.run_sql(sql)
        return df.to_dict('records')
```

#### API Endpoints:

**File: `app/main.py`**

```python
from app.services.sql_service import TextToSQLService

# Initialize service
sql_service = TextToSQLService(
    database_url=settings.DATABASE_URL,
    openai_api_key=settings.OPENAI_API_KEY
)

# Train once on startup
@app.on_event("startup")
def train_vanna():
    sql_service.train()

@app.post("/query/sql")
async def query_sql(question: str):
    """
    Generate and execute SQL from natural language
    For now, auto-executes (user approval added later)
    """
    # Generate SQL
    result = sql_service.generate_sql(question)
    sql = result['sql']
    
    # Execute SQL
    try:
        data = sql_service.execute_sql(sql)
        return {
            "question": question,
            "sql": sql,
            "results": data,
            "status": "success"
        }
    except Exception as e:
        return {
            "question": question,
            "sql": sql,
            "error": str(e),
            "status": "error"
        }
```

#### Testing:
- POST to `/query/sql` with: "How many customers?"
- Should return SQL + results
- Try 5-10 different questions

#### Deliverables:
- ✅ Vanna.ai connected to Supabase
- ✅ Trained on schema + 5 golden examples
- ✅ Can generate SQL from questions
- ✅ Can execute SQL and return results
- ✅ `/query/sql` endpoint working

---

## **WEEK 3: Document RAG + Query Routing**

### **Day 11-12: Document Search**

#### Goal:
Question → Search ChromaDB → Return relevant chunks

#### Implementation:

**File: `app/services/vector_service.py`**

```python
class VectorStore:
    # ... previous code ...
    
    async def search(self, query: str, top_k: int = 3) -> dict:
        """
        Search for relevant document chunks
        
        Args:
            query: User's question
            top_k: Number of results to return
        
        Returns:
            Dictionary with results and metadata
        """
        # Generate query embedding
        from app.services.embedding_service import generate_embeddings
        query_embedding = await generate_embeddings([query])
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return {
            'query': query,
            'chunks': chunks,
            'total_found': len(chunks)
        }
```

#### API Endpoint:

**File: `app/main.py`**

```python
@app.post("/query/search")
async def search_documents(query: str, top_k: int = 3):
    """Search for relevant document chunks"""
    results = await vector_store.search(query, top_k)
    return results
```

#### Testing:
- Upload a document with known content
- Search with related question
- Verify relevant chunks are returned

#### Deliverables:
- ✅ Can search ChromaDB with natural language
- ✅ Returns top 3 most relevant chunks
- ✅ `/query/search` endpoint working

---

### **Day 13-15: RAG Generation**

#### Goal:
Question → Search docs → Generate answer with GPT-4

#### Implementation:

**File: `app/services/rag_service.py`**

```python
import openai
from app.services.vector_service import VectorStore

class RAGService:
    def __init__(self, openai_api_key: str):
        self.client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.vector_store = VectorStore()
    
    async def generate_answer(self, question: str, top_k: int = 3) -> dict:
        """
        Full RAG pipeline: search + generate
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            Generated answer with sources
        """
        # Step 1: Search for relevant chunks
        search_results = await self.vector_store.search(question, top_k)
        chunks = search_results['chunks']
        
        # Step 2: Build context from chunks
        context = "\n\n".join([
            f"[Source: {chunk['metadata']['filename']}]\n{chunk['text']}"
            for chunk in chunks
        ])
        
        # Step 3: Create prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
        
        # Step 4: Generate answer
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Step 5: Format response
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "filename": chunk['metadata']['filename'],
                    "chunk_index": chunk['metadata']['chunk_index']
                }
                for chunk in chunks
            ],
            "chunks_used": len(chunks)
        }
```

#### API Endpoint:

**File: `app/main.py`**

```python
from app.services.rag_service import RAGService

rag_service = RAGService(openai_api_key=settings.OPENAI_API_KEY)

@app.post("/query/documents")
async def query_documents(question: str):
    """Query documents using RAG"""
    result = await rag_service.generate_answer(question)
    return result
```

#### Testing:
- Ask questions about uploaded documents
- Verify answers are accurate
- Check sources are cited

#### Deliverables:
- ✅ Full RAG pipeline working
- ✅ Generates answers from document context
- ✅ Includes source citations
- ✅ `/query/documents` endpoint working

---

### **Day 16-17: Query Routing**

#### Goal:
Automatically decide: SQL vs Documents vs Both

#### Implementation:

**File: `app/services/router_service.py`**

```python
class QueryRouter:
    """Simple rule-based router"""
    
    SQL_KEYWORDS = [
        'how many', 'count', 'total', 'sum', 'average',
        'list all', 'show all', 'find all',
        'revenue', 'sales', 'orders', 'customers',
        'maximum', 'minimum', 'highest', 'lowest'
    ]
    
    DOCUMENT_KEYWORDS = [
        'policy', 'procedure', 'guide', 'manual',
        'according to', 'document says', 'explain',
        'what is', 'definition', 'describe'
    ]
    
    @staticmethod
    def route(question: str) -> str:
        """
        Determine query type based on keywords
        
        Returns: 'SQL', 'DOCUMENTS', or 'HYBRID'
        """
        question_lower = question.lower()
        
        has_sql_keywords = any(kw in question_lower for kw in QueryRouter.SQL_KEYWORDS)
        has_doc_keywords = any(kw in question_lower for kw in QueryRouter.DOCUMENT_KEYWORDS)
        
        if has_sql_keywords and has_doc_keywords:
            return 'HYBRID'
        elif has_sql_keywords:
            return 'SQL'
        elif has_doc_keywords:
            return 'DOCUMENTS'
        else:
            # Default to documents for ambiguous queries
            return 'DOCUMENTS'
```

#### Unified Query Endpoint:

**File: `app/main.py`**

```python
from app.services.router_service import QueryRouter

@app.post("/query")
async def unified_query(question: str):
    """
    Main query endpoint - automatically routes to SQL or Documents
    """
    # Route the query
    query_type = QueryRouter.route(question)
    
    if query_type == 'SQL':
        # Use SQL service
        result = sql_service.generate_sql(question)
        sql = result['sql']
        data = sql_service.execute_sql(sql)
        
        return {
            "question": question,
            "type": "SQL",
            "sql": sql,
            "results": data
        }
    
    elif query_type == 'DOCUMENTS':
        # Use RAG service
        result = await rag_service.generate_answer(question)
        
        return {
            "question": question,
            "type": "DOCUMENTS",
            "answer": result['answer'],
            "sources": result['sources']
        }
    
    elif query_type == 'HYBRID':
        # Use both
        sql_result = sql_service.generate_sql(question)
        doc_result = await rag_service.generate_answer(question)
        
        return {
            "question": question,
            "type": "HYBRID",
            "sql_results": {
                "sql": sql_result['sql'],
                "data": sql_service.execute_sql(sql_result['sql'])
            },
            "document_results": {
                "answer": doc_result['answer'],
                "sources": doc_result['sources']
            }
        }
```

#### Testing:
- "How many customers?" → Should route to SQL
- "What is our return policy?" → Should route to DOCUMENTS
- "Show sales and explain our pricing policy" → Should route to HYBRID

#### Deliverables:
- ✅ Router correctly identifies query type
- ✅ `/query` endpoint handles all types
- ✅ 80%+ routing accuracy on test questions

---

## **WEEK 4: SQL Safety + Basic Evaluation**

### **Day 18-19: User Approval for SQL**

#### Goal:
Show SQL to user before executing

#### Implementation:

**File: `app/services/sql_service.py`**

```python
import uuid
from typing import Dict

class TextToSQLService:
    def __init__(self, database_url: str, openai_api_key: str):
        # ... previous init code ...
        self.pending_queries: Dict[str, dict] = {}  # Store pending SQL queries
    
    def generate_sql_for_approval(self, question: str) -> dict:
        """
        Generate SQL but don't execute yet
        Return query_id for later approval
        """
        sql = self.vn.generate_sql(question=question)
        
        # Generate unique ID
        query_id = str(uuid.uuid4())
        
        # Store pending query
        self.pending_queries[query_id] = {
            'question': question,
            'sql': sql,
            'status': 'pending'
        }
        
        return {
            'query_id': query_id,
            'question': question,
            'sql': sql,
            'explanation': f"This SQL will retrieve data from your database",
            'status': 'pending_approval'
        }
    
    def execute_approved_query(self, query_id: str, approved: bool):
        """Execute query if approved by user"""
        if query_id not in self.pending_queries:
            return {'error': 'Query not found'}
        
        query_info = self.pending_queries[query_id]
        
        if not approved:
            return {
                'query_id': query_id,
                'status': 'rejected',
                'message': 'Query execution cancelled by user'
            }
        
        # Execute the SQL
        try:
            sql = query_info['sql']
            results = self.execute_sql(sql)
            
            # Clean up
            del self.pending_queries[query_id]
            
            return {
                'query_id': query_id,
                'question': query_info['question'],
                'sql': sql,
                'results': results,
                'status': 'executed'
            }
        except Exception as e:
            return {
                'query_id': query_id,
                'error': str(e),
                'status': 'error'
            }
```

#### API Endpoints:

**File: `app/main.py`**

```python
@app.post("/query/sql/generate")
async def generate_sql_for_approval(question: str):
    """
    Step 1: Generate SQL and return for user approval
    """
    result = sql_service.generate_sql_for_approval(question)
    return result

@app.post("/query/sql/execute")
async def execute_sql_with_approval(query_id: str, approved: bool):
    """
    Step 2: Execute SQL after user approval
    
    Args:
        query_id: ID from generate endpoint
        approved: True to execute, False to cancel
    """
    result = sql_service.execute_approved_query(query_id, approved)
    return result
```

#### Update Main Query Endpoint:

```python
@app.post("/query")
async def unified_query(question: str, auto_approve_sql: bool = False):
    """
    Main query endpoint with SQL approval
    
    Args:
        auto_approve_sql: If True, execute SQL without approval (for testing)
    """
    query_type = QueryRouter.route(question)
    
    if query_type == 'SQL':
        if auto_approve_sql:
            # Direct execution (for testing)
            result = sql_service.generate_sql(question)
            sql = result['sql']
            data = sql_service.execute_sql(sql)
            return {
                "type": "SQL",
                "sql": sql,
                "results": data
            }
        else:
            # Return for approval
            result = sql_service.generate_sql_for_approval(question)
            return {
                "type": "SQL",
                "requires_approval": True,
                **result
            }
    
    # ... documents and hybrid logic unchanged ...
```

#### Testing:
1. POST `/query/sql/generate` with question
2. Get back query_id and SQL
3. POST `/query/sql/execute` with query_id and approved=true
4. Verify SQL executes

#### Deliverables:
- ✅ SQL generation doesn't auto-execute
- ✅ User sees SQL before execution
- ✅ Two-step approval process working
- ✅ Can approve or reject SQL queries

---

### **Day 20-21: RAGAS Evaluation**

#### Goal:
Measure system quality with RAGAS on 10 test queries

#### Test Dataset Creation:

**File: `tests/test_queries.json`**

```json
{
  "queries": [
    {
      "id": 1,
      "type": "sql",
      "question": "How many customers do we have?",
      "ground_truth": "We have 100 customers in total",
      "expected_sql": "SELECT COUNT(*) FROM customers"
    },
    {
      "id": 2,
      "type": "sql",
      "question": "What is the total revenue from delivered orders?",
      "ground_truth": "Total revenue from delivered orders",
      "expected_sql": "SELECT SUM(total_amount) FROM orders WHERE status = 'Delivered'"
    },
    {
      "id": 3,
      "type": "sql",
      "question": "How many orders per customer segment?",
      "ground_truth": "Number of orders grouped by customer segment",
      "expected_sql": "SELECT c.segment, COUNT(o.id) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.segment"
    },
    {
      "id": 4,
      "type": "document",
      "question": "What is our company's return policy?",
      "ground_truth": "Our return policy allows returns within 30 days...",
      "expected_source": "policy_document.pdf"
    },
    {
      "id": 5,
      "type": "document",
      "question": "Explain the onboarding process for new customers",
      "ground_truth": "The onboarding process consists of...",
      "expected_source": "onboarding_guide.docx"
    },
    {
      "id": 6,
      "type": "document",
      "question": "What products are mentioned in the catalog?",
      "ground_truth": "The catalog mentions products including...",
      "expected_source": "product_catalog.pdf"
    },
    {
      "id": 7,
      "type": "hybrid",
      "question": "Show me total sales and explain our pricing strategy",
      "ground_truth": "Combined SQL results and document explanation",
      "expected_sources": ["orders table", "pricing_policy.pdf"]
    },
    {
      "id": 8,
      "type": "sql",
      "question": "List all products in Electronics category",
      "ground_truth": "List of electronics products",
      "expected_sql": "SELECT * FROM products WHERE category = 'Electronics'"
    },
    {
      "id": 9,
      "type": "document",
      "question": "What are the customer support contact methods?",
      "ground_truth": "Support contact methods include...",
      "expected_source": "support_guide.pdf"
    },
    {
      "id": 10,
      "type": "sql",
      "question": "What is the average order value?",
      "ground_truth": "Average order value calculation",
      "expected_sql": "SELECT AVG(total_amount) FROM orders"
    }
  ]
}
```

#### Evaluation Script:

**File: `evaluate.py`**

```python
import json
import asyncio
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

async def run_evaluation():
    """Run RAGAS evaluation on test queries"""
    
    # Load test queries
    with open('tests/test_queries.json', 'r') as f:
        test_data = json.load(f)
    
    # Run queries through system and collect results
    results = []
    
    for query in test_data['queries']:
        # Query the system
        response = await query_system(query['question'])
        
        # Collect data for RAGAS
        results.append({
            'question': query['question'],
            'answer': response['answer'],
            'contexts': response.get('contexts', []),
            'ground_truth': query['ground_truth']
        })
    
    # Convert to RAGAS dataset format
    dataset = Dataset.from_dict({
        'question': [r['question'] for r in results],
        'answer': [r['answer'] for r in results],
        'contexts': [r['contexts'] for r in results],
        'ground_truth': [r['ground_truth'] for r in results]
    })
    
    # Run RAGAS evaluation
    evaluation_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ]
    )
    
    print("=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness: {evaluation_result['faithfulness']:.3f}")
    print(f"Answer Relevancy: {evaluation_result['answer_relevancy']:.3f}")
    print("=" * 50)
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_result, f, indent=2)
    
    return evaluation_result

if __name__ == "__main__":
    asyncio.run(run_evaluation())
```

#### Run Evaluation:

```bash
python evaluate.py
```

#### Deliverables:
- ✅ 10 test queries created
- ✅ RAGAS evaluation script working
- ✅ Faithfulness and Answer Relevancy scores
- ✅ Results saved to evaluation_results.json

**Target Scores:**
- Faithfulness: > 0.7
- Answer Relevancy: > 0.8

---

## **WEEK 5: Monitoring + Docker Deployment**

### **Day 22-23: OPIK Monitoring**

#### Goal:
Track all requests and performance metrics

#### Implementation:

**File: `app/main.py`**

```python
from opik import track
import opik

# Initialize OPIK
opik.configure(api_key=settings.OPIK_API_KEY)

@app.post("/upload")
@track(name="document_upload")
async def upload_document(file: UploadFile = File(...)):
    # ... existing code ...
    pass

@app.post("/query")
@track(name="unified_query")
async def unified_query(question: str):
    # ... existing code ...
    pass

@app.post("/query/sql/generate")
@track(name="sql_generation")
async def generate_sql_for_approval(question: str):
    # ... existing code ...
    pass
```

#### Track Custom Metrics:

```python
from opik import track_metric

async def generate_answer(self, question: str):
    import time
    start = time.time()
    
    # ... RAG pipeline ...
    
    duration = time.time() - start
    track_metric("rag_latency", duration)
    
    return result
```

#### View Dashboard:
- Visit opik.ai
- Navigate to your project
- View metrics: request count, latency, errors

#### Deliverables:
- ✅ OPIK decorators on key functions
- ✅ Requests visible in OPIK dashboard
- ✅ Latency metrics tracked
- ✅ Error tracking enabled

---

### **Day 24-25: Docker Deployment**

#### Goal:
Single Docker image that runs the entire application

#### Dockerfile:

**File: `Dockerfile`**

```dockerfile
# Use Python 3.12 slim base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Unstructured.io
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p ./data/uploads ./data/chromadb

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Environment File:

**File: `.env.example`**

```env
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Supabase/PostgreSQL
DATABASE_URL=postgresql://user:password@host:port/database

# OPIK
OPIK_API_KEY=your-opik-key

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

#### Build and Run:

**Build the image:**
```bash
docker build -t rag-text-to-sql:v1 .
```

**Run the container:**
```bash
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -v $(pwd)/data/chromadb:/app/data/chromadb \
  -v $(pwd)/data/uploads:/app/data/uploads \
  --env-file .env \
  rag-text-to-sql:v1
```

**View logs:**
```bash
docker logs -f rag-api
```

**Stop container:**
```bash
docker stop rag-api
```

**Start container:**
```bash
docker start rag-api
```

#### Testing Docker Deployment:
1. Build image
2. Run container
3. Visit http://localhost:8000/docs
4. Test all endpoints
5. Verify data persists after restart

#### Deliverables:
- ✅ Working Dockerfile
- ✅ .env.example template
- ✅ Can build Docker image
- ✅ Can run with single docker run command
- ✅ Data persists in volumes
- ✅ All features work in Docker

---

## 📊 Final Feature Checklist

### Core Features (INCLUDED):
- ✅ Upload PDF/DOCX/CSV/JSON documents
- ✅ Automatic document parsing (Unstructured.io)
- ✅ Text chunking (512 tokens, 50 overlap)
- ✅ OpenAI embeddings
- ✅ ChromaDB vector storage (single collection)
- ✅ Text-to-SQL with Vanna.ai
- ✅ User approval required for SQL execution
- ✅ Document RAG with GPT-4
- ✅ Automatic query routing (SQL vs Documents)
- ✅ 3-table database schema in Supabase
- ✅ RAGAS evaluation (10 test queries)
- ✅ OPIK monitoring
- ✅ Single Docker image deployment
- ✅ FastAPI with auto-documentation

### Excluded (Removed for Simplicity):
- ❌ Multiple document parsers (only Unstructured.io)
- ❌ Separate ChromaDB collections per doc type
- ❌ Advanced chunking strategies
- ❌ MMR/re-ranking
- ❌ Conversation memory
- ❌ Comprehensive test suites
- ❌ Performance optimization
- ❌ CI/CD pipeline
- ❌ Authentication/authorization
- ❌ Rate limiting
- ❌ Complex database (5+ tables)
- ❌ 50+ test queries

---

## 📈 Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Document Upload** | Works | Any PDF/DOCX/CSV/JSON |
| **Document Retrieval** | Top-3 relevant chunks | For document queries |
| **SQL Generation** | 70%+ accuracy | On test queries |
| **SQL Approval** | Required | No auto-execution |
| **Query Routing** | 80%+ correct | SQL vs Document classification |
| **RAGAS Faithfulness** | > 0.7 | No hallucinations |
| **RAGAS Answer Relevancy** | > 0.8 | Answers the question |
| **Response Time** | < 15 seconds | End-to-end query |
| **Docker Deployment** | Single command | `docker run` works |

---

## 🚀 Getting Started

### Prerequisites:
- Python 3.12+
- Docker Desktop
- Supabase account
- OpenAI API key
- OPIK account

### Quick Start:

```bash
# 1. Clone repository
git clone <your-repo>
cd project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run application
uvicorn app.main:app --reload

# 6. Visit documentation
open http://localhost:8000/docs
```

### Docker Quick Start:

```bash
# 1. Build image
docker build -t rag-text-to-sql .

# 2. Run container
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  rag-text-to-sql

# 3. View logs
docker logs -f <container-id>
```

---

## 🔧 API Endpoints Reference

### Document Management:
- `POST /upload` - Upload a document
- `GET /documents` - List all documents

### Query Endpoints:
- `POST /query` - Main query endpoint (auto-routes)
- `POST /query/sql/generate` - Generate SQL (step 1)
- `POST /query/sql/execute` - Execute SQL after approval (step 2)
- `POST /query/documents` - Query documents only

### Utility:
- `GET /health` - Health check
- `GET /info` - System information

---

## 📝 Common Issues & Solutions

### Issue: Unstructured.io fails to parse PDF
**Solution:** Check if system dependencies are installed (poppler, tesseract)

### Issue: ChromaDB not persisting
**Solution:** Verify volume mount: `-v $(pwd)/data/chromadb:/app/data/chromadb`

### Issue: OpenAI rate limits
**Solution:** Add retry logic with exponential backoff (already included)

### Issue: Vanna generates incorrect SQL
**Solution:** Add more training examples specific to your schema

### Issue: Low RAGAS scores
**Solution:** Improve chunking strategy or increase top_k retrieval

---

## 🎯 Next Steps After MVP

Once core system is working, consider:

1. **Add conversation memory** for follow-up questions
2. **Implement caching** for repeated queries
3. **Add authentication** (API keys)
4. **Separate collections** per document type
5. **Advanced error handling** and validation
6. **Performance optimization** (async processing)
7. **CI/CD pipeline** (GitHub Actions)
8. **Load testing** and scaling
9. **More comprehensive test suite**
10. **Production deployment** (AWS/GCP/Azure)

---

## 📚 Additional Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com
- **ChromaDB Docs:** https://docs.trychroma.com
- **Vanna.ai Docs:** https://vanna.ai/docs
- **RAGAS Docs:** https://docs.ragas.io
- **OPIK Docs:** https://www.opik.ai/docs
- **OpenAI API:** https://platform.openai.com/docs

---

## ✅ Project Completion Checklist

### Week 1:
- [ ] Development environment setup
- [ ] FastAPI app running
- [ ] Document upload working
- [ ] ChromaDB storing embeddings

### Week 2:
- [ ] Database created in Supabase
- [ ] Sample data generated
- [ ] Vanna.ai trained
- [ ] SQL generation working

### Week 3:
- [ ] Document search working
- [ ] RAG generation working
- [ ] Query routing implemented
- [ ] Main /query endpoint functional

### Week 4:
- [ ] SQL approval flow added
- [ ] 10 test queries created
- [ ] RAGAS evaluation complete
- [ ] Scores documented

### Week 5:
- [ ] OPIK monitoring active
- [ ] Dockerfile created
- [ ] Docker image builds
- [ ] Full system runs in Docker
- [ ] README documentation complete

---

**Project Status:** Ready for Implementation  
**Estimated Completion:** 4-5 weeks  
**Next Action:** Begin Week 1, Day 1 setup

---

*This is a living document. Update as implementation progresses.*
