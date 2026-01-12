#  Multi-Agent Customer Support AI System

> An intelligent, LangGraph-powered customer support system with personalized product recommendations, session persistence, and supervisor escalation dashboard.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Agent Workflows](#agent-workflows)
- [Database Schema](#database-schema)
- [ML Recommendation System](#ml-recommendation-system)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

##  Overview

This project implements a **multi-agent customer support system** that intelligently routes customer queries to specialized agents:

- **Support Agent**: Handles FAQs, policies, and general inquiries
- **Query Parser + Recommendation Agent**: Provides personalized product recommendations using ML
- **Escalation Agent**: Identifies high-priority issues and routes them to human supervisors

The system features **session persistence**, **streaming responses**, and a **real-time supervisor dashboard** for managing escalated queries.

---

##  Features

### Core Functionality
-  **Multi-Agent Orchestration** using LangGraph
-  **Intelligent Query Routing** based on intent
-  **ML-Powered Product Recommendations** (LightGBM + Sentence Transformers)
-  **Session Persistence** (survives page reloads)
-  **Real-time Streaming Responses** (SSE)
-  **Customer Profile & Purchase History Integration**
-  **Supervisor Escalation Dashboard**

### User Experience
-  Guest & Registered User Support
-  Personalized Greetings & Recommendations
-  Filter-based Product Search (brand, price, gender, category)
-  Conversation History Tracking
-  Mobile-Responsive Design

### Security & Performance
-  No localStorage usage (artifact-safe)
-  Session validation middleware
-  Error handling & graceful degradation
-  NO LLM hallucination (ML catalog as source of truth)

---

##  Architecture

<img width="5654" height="8192" alt="agentic workflow" src="https://github.com/user-attachments/assets/47deceae-c3cb-41b6-a7ec-6a80d0c67b04" />


##  Tech Stack

### Backend
- **Python 3.9+**
- **Flask** - Web framework
- **LangGraph** - Agent orchestration
- **LangChain** - LLM integration
- **OpenAI GPT-4o-mini** - Natural language understanding
- **LightGBM** - ML recommendation model
- **Sentence Transformers** - Semantic search
- **SQLite** - Database
- **FAISS** - Vector store for policy retrieval

### Frontend
- **React 18** (via CDN)
- **Tailwind CSS** (via CDN)
- **Vanilla JavaScript** - No build process required

### ML Pipeline
- **scikit-learn** - Preprocessing
- **pandas** - Data manipulation
- **numpy** - Numerical operations

---

##  Installation

### Prerequisites
```bash
Python 3.9 or higher
Node.js (optional, not required)
```

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/customer-support-ai.git
cd customer-support-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
flask==3.0.0
flask-cors==4.0.0
langchain==0.1.0
langchain-openai==0.0.2
langgraph==0.0.20
pydantic==2.5.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
lightgbm==4.1.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
python-dotenv==1.0.0
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Initialize Database
```bash
python backend/Database/database.py
```

This will:
- Create `customer_support.db`
- Set up tables
- Seed sample customer data

### 6. Build Vector Store (Policy Documents)
```bash
cd backend
python build_vector_store.py
```

### 7. Train ML Recommendation Model
```bash
cd recommendation_system
python train.py
```

This will generate:
- `recommender_model.pkl`
- `feature_builder.pkl`

---

##  Configuration

### Update File Paths

**In `main.py`:**
```python
# Line 20-22
RECOMMENDER_MODEL = joblib.load(r"YOUR_PATH/recommender_model.pkl")
FEATURE_BUILDER = joblib.load(r"YOUR_PATH/feature_builder.pkl")
PRODUCT_CATALOG = pd.read_csv(r"YOUR_PATH/myntra_products_catalog_v2.csv")
```

**In `database.py`:**
```python
# Line 6
DB_PATH = "YOUR_PATH/customer_support.db"
```

**In `frontend/index.html`:**
```javascript
// Line 158 (if different from localhost)
const response = await fetch("http://YOUR_SERVER:5000/chat", {
```
--- 

### Add category column to dataset 
'''bash
cd data
python category.p
'''

##  Running the Application

### 1. Start Backend Server
```bash
cd backend
python main.py
```

Server starts on: `http://localhost:5000`

### 2. Open Frontend
```bash
# Navigate to frontend directory
cd frontend

# Open index.html in browser
# OR use a simple server:
python -m http.server 8000
```

Frontend available at: `http://localhost:8000`

### 3. Open Supervisor Dashboard
```bash
cd supervisor_dashboard
# Open supervisor.html in browser
```

Dashboard available at: `file:///path/to/supervisor.html`

---


##  API Documentation

### Authentication

#### POST `/login`
**Body:**
```json
{
  "email": "john@example.com"
}
```

**Response:**
```json
{
  "success": true,
  "customer_id": "CUST001",
  "email": "john@example.com",
  "profile": { ... },
  "insights": { ... },
  "conversation_history": [ ... ]
}
```

#### POST `/logout`
**Body:**
```json
{
  "customer_id": "CUST001"
}
```

---

### Chat

#### POST `/chat`
**Body:**
```json
{
  "message": "Show me Nike shoes under 5000",
  "email": "john@example.com"  // Optional for guests
}
```

**Response:** Server-Sent Events (SSE)
```javascript
data: {"type": "agent_start", "agent": "Recommendation Agent"}
data: {"type": "token", "content": "Here are "}
data: {"type": "token", "content": "some options..."}
data: {"type": "done", "metadata": {...}}
```

---

### Customer Operations

#### GET `/customer/<customer_id>`
Get customer profile and purchase history.

#### POST `/customer`
Create new customer.

#### POST `/purchase`
Add purchase record.

---

### Escalation Management (Supervisor)

#### GET `/escalations`
Get all escalations.

**Response:**
```json
{
  "escalations": [ ... ],
  "total": 5,
  "pending": 3,
  "resolved": 2
}
```

#### POST `/escalations/<escalation_id>/resolve`
Mark escalation as resolved.

**Body:**
```json
{
  "response": "Issue resolved via phone call"
}
```

---

##  Agent Workflows

### 1. Support Agent Flow
```
User Query → Orchestrator → Support Agent
                ↓
    Retrieve Policy from Vector Store
                ↓
    Format Response with Customer Context
                ↓
          Return Answer
```

### 2. Recommendation Agent Flow
```
User Query → Orchestrator → Query Parser
                ↓
    Extract Filters (brand, price, gender)
                ↓
         Recommendation Agent
                ↓
    ML Model Ranks Products from Catalog
                ↓
    LLM Formats Output (NO Hallucination)
                ↓
          Return Products
```

### 3. Escalation Agent Flow
```
User Query → Orchestrator → Escalation Agent
                ↓
    Analyze Query + Customer Value
                ↓
    Calculate Confidence Score
                ↓
    If >= 70% → Create Escalation Record
                ↓
    Notify User + Send to Supervisor Dashboard
```

---

## 🗄️ Database Schema

### Customers Table
```sql
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    gender TEXT,
    preferred_categories TEXT,
    preferred_brands TEXT,
    budget_range TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Purchase History Table
```sql
CREATE TABLE purchase_history (
    purchase_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    product_name TEXT NOT NULL,
    product_brand TEXT,
    category TEXT,
    price REAL NOT NULL,
    purchase_date TIMESTAMP,
    rating INTEGER,
    review TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

### Conversations Table
```sql
CREATE TABLE conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    message_role TEXT NOT NULL,
    message_content TEXT NOT NULL,
    agent_name TEXT,
    metadata TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

---

##  ML Recommendation System

### Architecture
1. **Feature Engineering** (`preprocessing.py`)
   - Price normalization
   - Brand/Color encoding
   - TF-IDF text features
   - Sentence embeddings (all-MiniLM-L6-v2)

2. **Model Training** (`train.py`)
   - LightGBM classifier
   - Semantic similarity as pseudo-labels
   - Ranking-based evaluation (Precision@K, Recall@K)


### Key Features
-  **NO LLM Hallucination**: Only real products from catalog
-  **Personalized Ranking**: Based on purchase history
-  **Dynamic Filtering**: Price, brand, category, gender
-  **Semantic Search**: Natural language understanding

---

##  Troubleshooting

### Issue: Session Lost on Page Reload
**Solution:** Ensure `sessionLoadedRef` is implemented correctly in `index.html` (lines 74-102).

### Issue: ML Models Not Loading
**Solution:**
1. Check file paths in `main.py` (lines 60-75)
2. Verify `.pkl` files exist in `recommendation_system/` directory
3. Run `train_lightgbm.py` to regenerate models

### Issue: Database Connection Error
**Solution:**
1. Check `DB_PATH` in `database.py`
2. Run `python database.py` to initialize
3. Verify `customer_support.db` file exists

### Issue: CORS Errors
**Solution:** Ensure Flask CORS is properly configured:
```python
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
```

### Issue: Streaming Not Working
**Solution:**
1. Check SSE headers in `/chat` endpoint
2. Verify `create_sse_message()` format
3. Test with: `curl -N http://localhost:5000/chat -X POST -d '{"message":"test"}'`

---

##  Testing

### Test Sample Users
```
john@example.com  - 3 purchases, ₹13,997 spent
jane@example.com  - 2 purchases, ₹11,998 spent
alex@example.com  - 0 purchases
```

### Test Queries
```
"Show me Adidas shoes under 5000"
"I need formal shoes for men"
"Suggest something similar to what I bought"
"What is your return policy?"
"I want a refund for my order!"  (triggers escalation)
```

---

##  Project Structure

```
customer-support-ai/
├── backend/
│   ├── main.py                 # Flask server + agents
│   ├── Database/
│   │   └── database.py         # SQLite operations
│   ├── recommendation_system/
│   │   ├── train.py   # Model training
│   │   ├── preprocessing.py    # Feature engineering
│   │     
│   ├── vector_store.py         # FAISS policy store
│   └── build_vector_store.py   # Policy indexing
├── frontend/
│   └── index.html              # React chat interface
├── supervisor_dashboard/
│   └── queries.html         # Escalation management
├── data/
│   └── myntra_products_catalog.csv
├── .env
├── requirements.txt
└── README.md
```

---

##  Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---
