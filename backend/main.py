from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Generator, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import add
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import time
import json
import sys
sys.path.append(r"path to database.py")
from Database.database import CustomerDatabase

load_dotenv()

# FLASK APP SETUP
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app) # CORS is less critical now as we are on the same origin

# DATABASE INITIALIZATION

customer_db = CustomerDatabase()
print("Using DB:", os.path.abspath(customer_db.db_path))

# ML MODELS INITIALIZATION

RECOMMENDER_MODEL = None
FEATURE_BUILDER = None
PRODUCT_CATALOG = None
PRODUCT_FEATURES = None
TEXT_MODEL = None
PRODUCT_EMBEDDINGS = None
ML_LOADED = False


def initialize_ml_models():
    """Initialize ML models at startup"""
    global RECOMMENDER_MODEL, FEATURE_BUILDER, PRODUCT_CATALOG
    global PRODUCT_FEATURES, TEXT_MODEL, PRODUCT_EMBEDDINGS, ML_LOADED
    
    try:
        RECOMMENDER_MODEL = joblib.load(r" path to recommender_model.pkl")
        FEATURE_BUILDER = joblib.load(r"path to feature builder.pkl")
        PRODUCT_CATALOG = pd.read_csv(r"path to dataset)
        
        print(" Precomputing product features...")
        PRODUCT_FEATURES = FEATURE_BUILDER.transform_products(PRODUCT_CATALOG)
        
        TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        product_texts = (PRODUCT_CATALOG['ProductName'] + " " + PRODUCT_CATALOG['Description']).tolist()
        PRODUCT_EMBEDDINGS = TEXT_MODEL.encode(product_texts, show_progress_bar=False)
        
        print("ML models and catalog loaded successfully")
        ML_LOADED = True
    except Exception as e:
        print(f"Could not load ML models: {e}")
        ML_LOADED = False


try:
    from vector_store import search
except ImportError:
    print("Warning: vector_store module not found. Using mock search.")
    def search(query: str, k: int = 3):
        return [
            {"text": "Our return policy allows returns within 30 days.", "score": 0.85},
            {"text": "We offer free shipping on orders over $50.", "score": 0.72},
            {"text": "Customer support available 24/7.", "score": 0.68}
        ]


# PYDANTIC MODELS & STATE

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    task: str
    next_agent: str
    current_agent: str
    iteration: int
    recommendations: str
    results: str
    confidence_score: float
    escalated_queries: Annotated[List[str], add]
    agents_executed: Annotated[List[str], add]
    needs_clarification: bool
    clarification_question: str
    clarification_context: str
    customer_id: str
    customer_context: Dict
    parsed_filters: Dict


class QueryFilters(BaseModel):
    """Structured output for query parsing"""
    product_type: Optional[str] = Field(default=None, description="Type of product: shoes, shirt, dress, pants, etc.")
    brand: Optional[str] = Field(default=None, description="Specific brand requested: Nike, Adidas, Puma, etc.")
    gender: Optional[str] = Field(default=None, description="Men, Women, or Unisex")
    color: Optional[str] = Field(default=None, description="Preferred color")
    min_price: Optional[float] = Field(default=None, description="Minimum price in INR")
    max_price: Optional[float] = Field(default=None, description="Maximum price in INR")
    category: Optional[str] = Field(default=None, description="Product category: Shoes, Topwear, Bottomwear, Dress, etc.")
    style: Optional[str] = Field(default=None, description="Style preference: casual, formal, sports, running, etc.")
    use_purchase_history: bool = Field(default=False, description="Should we base recommendations on past purchases?")
    reasoning: str = Field(default="", description="Why these filters were chosen")


class Escalation_Schema(BaseModel):
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="No reasoning provided")


class Clarification_Schema(BaseModel):
    needs_clarification: bool = Field()
    clarification_question: str = Field(default="")
    reasoning: str = Field(default="")


# HELPER FUNCTIONS

def get_customer_context(customer_id: str) -> Dict:
    """Get comprehensive customer context for recommendations"""
    
    customer_profile = customer_db.get_customer(customer_id)
    if not customer_profile:
        return {
            'profile': None,
            'purchase_history': [],
            'insights': {},
            'conversation_history': []
        }
    
    purchase_history = customer_db.get_purchase_history(customer_id, limit=10)
    insights = customer_db.get_customer_insights(customer_id)
    conversation_history = customer_db.get_conversation_history(customer_id, limit=5)
    
    return {
        'profile': customer_profile,
        'purchase_history': purchase_history,
        'insights': insights,
        'conversation_history': conversation_history
    }


def format_customer_context_for_llm(context: Dict) -> str:
    """Format customer context for LLM consumption"""
    
    if not context.get('profile'):
        return "No customer profile available."
    
    profile = context['profile']
    insights = context['insights']
    purchases = context['purchase_history']
    
    formatted = f"""
**Customer Profile:**
- Name: {profile['name']}
- Gender: {profile['gender']}
- Preferred Categories: {profile.get('preferred_categories', 'Not set')}
- Preferred Brands: {profile.get('preferred_brands', 'Not set')}
- Budget Range: {profile.get('budget_range', 'Not set')}

**Purchase History Insights:**
- Total Purchases: {insights['total_purchases']}
- Total Spent: ₹{insights['total_spent']:.2f}
- Avg Purchase Value: ₹{insights['avg_purchase_value']:.2f}
- Favorite Brands: {', '.join(insights['favorite_brands']) if insights['favorite_brands'] else 'None'}
- Favorite Categories: {', '.join(insights['favorite_categories']) if insights['favorite_categories'] else 'None'}

**Recent Purchases:**
"""
    
    for i, purchase in enumerate(purchases[:5], 1):
        formatted += f"{i}. {purchase['product_name']} by {purchase['product_brand']} - ₹{purchase['price']}\n"
    
    return formatted.strip()


# AGENT FUNCTIONS

def orchestrator(state: State) -> State:
    """Routes queries to appropriate agents"""
    
    orchestrator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    task = state.get('task', '')
    current_agent = state.get('current_agent', '')
    iteration = state.get('iteration', 0)
    results = state.get('results', '')
    recommendations = state.get('recommendations', '')
    confidence_score = state.get('confidence_score', 0.0)
    needs_clarification = state.get('needs_clarification', False)
    clarification_question = state.get('clarification_question', '')
    
    print(f"\n{'='*80}")
    print(f"🎯 ORCHESTRATOR - Iteration {iteration}")
    print(f"{'='*80}")
    
    # Handle clarification requests
    if needs_clarification and clarification_question:
        print(f"❓ Agent needs clarification: {clarification_question}")
        return {
            'messages': [AIMessage(content=f"[Orchestrator] Agent requesting clarification", name="orchestrator")],
            'next_agent': 'Finish',
            'current_agent': 'Orchestrator',
            'iteration': iteration + 1
        }
    
    # First iteration: Determine agent type
    if iteration == 0:
        prompt = f"""Analyze the query and determine which agent should handle it.

**Customer Query:** {task}

**Available Agents:**
1. Support Agent - For: policy questions, account issues, FAQs, returns, shipping
2. Query Parser + Recommendation Agent - For: product suggestions, shopping requests
3. Escalation Agent - For: billing disputes, complaints, urgent issues

Choose ONE agent. If it's a product recommendation request, choose "Query Parser".

Answer with ONLY the agent name: "Support Agent" OR "Query Parser" OR "Escalation Agent"."""
        
        response = orchestrator_llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        
        print(f" LLM Decision: {decision}")
        
        if 'query parser' in decision or 'parser' in decision:
            routing = 'Query Parser'
        elif 'support' in decision:
            routing = 'Support Agent'
        elif 'escalation' in decision:
            routing = 'Escalation Agent'
        else:
            routing = 'Support Agent'
            print(f" Could not parse decision, defaulting to Support Agent")
        
        print(f" Routing to: {routing}")
        
        return {
            'messages': [AIMessage(content=f"[Orchestrator] Routing to: {routing}", name="orchestrator")],
            'next_agent': routing,
            'current_agent': 'Orchestrator',
            'iteration': iteration + 1
        }
    
    # After Query Parser: Route to Recommendation Agent
    if current_agent == 'Query Parser':
        print(" Query parsed, routing to Recommendation Agent")
        return {
            'messages': [AIMessage(content="[Orchestrator] Getting recommendations...", name="orchestrator")],
            'next_agent': 'Recommendation Agent',
            'current_agent': 'Orchestrator',
            'iteration': iteration + 1
        }
    
    # Check if agent completed work
    # Decide if the workflow should finish or continue
    if iteration > 0:

    # If Query Parser just ran, we MUST go to Recommendation Agent
        if current_agent == "Query Parser":
            print(" Parsed filters ready, moving to Recommendation Agent")
            return {
            'messages': [AIMessage(content="[Orchestrator] Sending to Recommendation Agent", name="orchestrator")],
            'next_agent': 'Recommendation Agent',
            'current_agent': 'Orchestrator',
            'iteration': iteration + 1
        }

    # If Support Agent gave a response → done
        if current_agent == "Support Agent" and results:
            print(" Support response complete")
            return {
            'messages': [AIMessage(content="[Orchestrator] Support task complete", name="orchestrator")],
            'next_agent': 'Finish',
            'current_agent': 'Orchestrator',
            'iteration': iteration + 1
            }

    # If Recommendation Agent gave recommendations → done
        if current_agent == "Recommendation Agent" and recommendations:
            print(" Recommendations ready")
            return {
            'messages': [AIMessage(content="[Orchestrator] Recommendation complete", name="orchestrator")],
            'next_agent': 'Finish',
            'current_agent': 'Orchestrator',
            'iteration': iteration + 1
        }

    # If Escalation Agent ran → always finish
        if current_agent == "Escalation Agent":
            print(" Escalation handled")
            return {
            'messages': [AIMessage(content="[Orchestrator] Escalation handled", name="orchestrator")],
            'next_agent': 'Finish',
            'current_agent': 'Orchestrator',
            'iteration': iteration + 1
        }

    # Otherwise do NOT kill the workflow
        print(" Workflow not finished yet, waiting for next agent")
        return {
        'messages': [AIMessage(content="[Orchestrator] Continuing workflow", name="orchestrator")],
        'next_agent': state.get("next_agent", "Finish"),
        'current_agent': 'Orchestrator',
        'iteration': iteration + 1
    }



def query_parser_agent(state: State) -> State:
    """Extracts structured filters from natural language queries"""
    
    print(f"\n{'='*80}")
    print(f" QUERY PARSER AGENT EXECUTING")
    print(f"{'='*80}")
    
    parser_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    query = state.get('task', '')
    customer_context = state.get('customer_context') or {}
    
    # Build context for the LLM
    profile = customer_context.get('profile') or {}
    insights = customer_context.get('insights') or {}
    purchase_history = customer_context.get('purchase_history') or []
    
    customer_info = ""
    if profile:
        customer_info = f"""
**Customer Profile:**
- Name: {profile.get('name', 'Unknown')}
- Gender: {profile.get('gender', 'Unknown')}
- Past Purchases: {insights.get('total_purchases', 0)} items
- Favorite Brands: {', '.join(insights.get('favorite_brands', [])[:3]) or 'None'}
- Recent Purchases: {', '.join([p.get('product_name', '') for p in purchase_history[:3]]) or 'None'}
"""
    else:
        customer_info = """
**Customer Profile:**
- Guest user (no profile available)
- No purchase history
"""
    
    prompt = f"""Extract structured filters from the customer's product request.

**Customer Request:** {query}

{customer_info}

**Instructions:**
1. Extract explicit requirements (brand, type, price, color, etc.)
2. For GUEST users with vague queries, DO NOT infer - leave fields empty
3. For registered users, you can use profile data for missing fields
4. Set use_purchase_history=True ONLY for queries like "similar to what I bought"

**Examples:**

Query: "show me adidas running shoes under 5000"
→ brand: "Adidas", product_type: "shoes", style: "running", max_price: 5000, category: "Shoes"

Query: "suggest something similar to what I bought recently" (with profile)
→ use_purchase_history: True, reasoning: "Based on past purchases"

Query: "suggest something" (GUEST user)
→ reasoning: "Too vague, need more details"
(No filters set - will trigger clarification)

Query: "I need formal shoes" (GUEST user)
→ product_type: "shoes", style: "formal", category: "Shoes"

Query: "show me something from adidas" (with Men profile)
→ brand: "Adidas", gender: "Men", reasoning: "Using profile gender"

Query: "show me something from adidas" (GUEST user)
→ brand: "Adidas", reasoning: "Brand specified, but need gender/type"

Now parse this query:"""
    
    parser_llm_with_schema = parser_llm.with_structured_output(QueryFilters)
    
    try:
        parsed = parser_llm_with_schema.invoke([HumanMessage(content=prompt)])
        
        # Convert to dict and remove None values
        filters = {
            'product_type': parsed.product_type,
            'brand': parsed.brand,
            'gender': parsed.gender,
            'color': parsed.color,
            'min_price': parsed.min_price,
            'max_price': parsed.max_price,
            'category': parsed.category,
            'style': parsed.style,
            'use_purchase_history': parsed.use_purchase_history
        }
        
        filters = {k: v for k, v in filters.items() if v is not None}
        
        print(f" Parsed filters: {filters}")
        print(f" Reasoning: {parsed.reasoning}")
        
        return {
            'messages': [AIMessage(content=f"[Query Parser] Extracted filters: {filters}", name="query_parser")],
            'parsed_filters': filters,
            'current_agent': 'Query Parser',
            'agents_executed': ['Query Parser']
        }
    
    except Exception as e:
        print(f Parsing error: {e}")
        return {
            'messages': [AIMessage(content=f"[Query Parser] Error parsing query", name="query_parser")],
            'parsed_filters': {},
            'current_agent': 'Query Parser',
            'agents_executed': ['Query Parser']
        }


def support_agent(state: State) -> State:
    """Handles general customer support queries"""
    
    print(f"\n{'='*80}")
    print(f SUPPORT AGENT EXECUTING")
    print(f"{'='*80}")
    
    support_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    query = state.get('task', '')
    customer_context = state.get('customer_context', {})
    
    customer_info = ""
    if customer_context.get('profile'):
        customer_info = f"\n**Customer:** {customer_context['profile']['name']}"
    
    clarification_llm = support_llm.with_structured_output(Clarification_Schema)
    
    clarification_prompt = f"""Analyze if you need more information to answer this support query.

**Customer Query:** {query}
{customer_info}

**IMPORTANT: Only set needs_clarification to TRUE if:**
- Query mentions "my order" but no order number provided AND customer history doesn't help
- Query is extremely vague like just "help" with no context

**DO NOT request clarification for:**
- General policy questions
- Product information requests
- Queries that can be answered with general information

If answerable, set needs_clarification to False."""

    try:
        clarification_check = clarification_llm.invoke([HumanMessage(content=clarification_prompt)])
        
        if clarification_check.needs_clarification:
            print(f" Needs clarification: {clarification_check.clarification_question}")
            return {
                'messages': [AIMessage(content=f"[Support Agent] {clarification_check.clarification_question}", name="support_agent")],
                'needs_clarification': True,
                'clarification_question': clarification_check.clarification_question,
                'clarification_context': clarification_check.reasoning,
                'results': clarification_check.clarification_question,
                'current_agent': 'Support Agent',
                'agents_executed': ['Support Agent']
            }
    except Exception as e:
        print(f" Clarification check error: {e}")
    
    try:
        search_results = search(query, k=3)
        context = '\n'.join([f"- {result['text']}" for result in search_results])
        print(f" Retrieved {len(search_results)} relevant documents")
    except Exception as e:
        context = "No relevant context found."
        print(f" Search error: {e}")
    
    customer_context_str = format_customer_context_for_llm(customer_context)
    
    prompt = f"""You are a senior customer support specialist for a premium e-commerce platform.

**Customer Query:** {query}

{customer_context_str}

**Relevant Policy Information:**
{context}

**STRICT OUTPUT RULES:**
1. **PLAIN TEXT ONLY:** Do NOT use markdown formatting. No asterisks (**), no hashes (#), no bolding.
2. **NO PLACEHOLDERS:** Never use '[Your Name]' or '[Insert Date]'. Sign off simply as "Customer Support Team".
3. **TONE:** Be professional, concise, and empathetic. Avoid robotic phrases.
4. **FORMAT:** Use simple spacing for readability. Do not use bullet points characters that might break rendering; use hyphens (-) if a list is absolutely necessary.

**Instructions:**
- Answer the query directly based on the context provided.
- If the policy information answers the question, paraphrase it warmly.
- If the query is about a specific order and you have the history, refer to the product names naturally.

Your response:"""
    
    response = support_llm.invoke([HumanMessage(content=prompt)])
    
    print(f" Support response generated")
    
    return {
        'messages': [AIMessage(content=f"[Support Agent] {response.content}", name="support_agent")],
        'results': response.content,
        'current_agent': 'Support Agent',
        'agents_executed': ['Support Agent'],
        'needs_clarification': False
    }


def get_ml_recommendations(query: str, customer_context: Dict, 
                          filters: Dict, use_history: bool = False,
                          top_k: int = 5) -> List[Dict]:
    """ML recommendations with dynamic filters from Query Parser"""
    
    if not ML_LOADED:
        return []
    
    print(f"\n ML Engine: Processing with filters={filters}, use_history={use_history}")
    
    profile = customer_context.get('profile', {})
    insights = customer_context.get('insights', {})
    purchase_history = customer_context.get('purchase_history', [])
    
    # Start with all products
    valid_mask = np.ones(len(PRODUCT_CATALOG), dtype=bool)
    
    # Apply dynamic filters from Query Parser
    if filters.get('gender'):
        gender_mask = (
            (PRODUCT_CATALOG['Gender'] == filters['gender']) | 
            (PRODUCT_CATALOG['Gender'] == 'Unisex')
        )
        valid_mask &= gender_mask
        print(f"  ✓ Gender={filters['gender']}: {gender_mask.sum()} products")
    
    if filters.get('brand'):
        brand_mask = PRODUCT_CATALOG['ProductBrand'].str.contains(
            filters['brand'], case=False, na=False
        )
        valid_mask &= brand_mask
        print(f"  ✓ Brand={filters['brand']}: {brand_mask.sum()} products")
    
    if filters.get('category'):
        category_mask = PRODUCT_CATALOG['Category'] == filters['category']
        valid_mask &= category_mask
        print(f"  ✓ Category={filters['category']}: {category_mask.sum()} products")
    
    if filters.get('max_price'):
        price_mask = PRODUCT_CATALOG['Price (INR)'] <= filters['max_price']
        valid_mask &= price_mask
        print(f"  ✓ Max price=₹{filters['max_price']}: {price_mask.sum()} products")
    
    if filters.get('min_price'):
        price_mask = PRODUCT_CATALOG['Price (INR)'] >= filters['min_price']
        valid_mask &= price_mask
        print(f"  ✓ Min price=₹{filters['min_price']}: {price_mask.sum()} products")
    
    print(f"📊 After filtering: {valid_mask.sum()} products remain")
    
    if valid_mask.sum() == 0:
        print(" No products after strict filters. Relaxing gender...")
        
        # Remove gender constraint
        relaxed_mask = np.ones(len(PRODUCT_CATALOG), dtype=bool)
        
        if filters.get('brand'):
            relaxed_mask &= PRODUCT_CATALOG['ProductBrand'].str.contains(filters['brand'], case=False, na=False)
        if filters.get('category'):
            relaxed_mask &= PRODUCT_CATALOG['Category'] == filters['category']
        if filters.get('max_price'):
            relaxed_mask &= PRODUCT_CATALOG['Price (INR)'] <= filters['max_price']
        
        if relaxed_mask.sum() > 0:
            valid_mask = relaxed_mask
        else:
            print(" Still no products after relaxing filters")
            return []
    
    # Semantic search
    query_embedding = TEXT_MODEL.encode([query])[0]
    similarities = cosine_similarity([query_embedding], PRODUCT_EMBEDDINGS)[0]
    
    # Preference boosting
    combined_scores = similarities.copy()
    
    # If use_history=True, boost products similar to past purchases
    if use_history and purchase_history:
        print(" Boosting based on purchase history...")
        purchased_categories = [p.get('category') for p in purchase_history if p.get('category')]
        purchased_brands = [p.get('product_brand') for p in purchase_history if p.get('product_brand')]
        
        for idx in range(len(PRODUCT_CATALOG)):
            if PRODUCT_CATALOG.iloc[idx]['Category'] in purchased_categories:
                combined_scores[idx] += 0.2
            if PRODUCT_CATALOG.iloc[idx]['ProductBrand'] in purchased_brands:
                combined_scores[idx] += 0.15
    
    # Boost favorite brands if no specific brand requested
    if not filters.get('brand') and insights.get('favorite_brands'):
        for brand in insights['favorite_brands']:
            brand_mask = PRODUCT_CATALOG['ProductBrand'] == brand
            combined_scores[brand_mask.values] += 0.15
            print(f" Boosting favorite brand: {brand}")
    
    # Apply mask
    combined_scores[~valid_mask] = -1
    
    # Deprioritize repurchases
    purchased_ids = {str(p['product_id']) for p in purchase_history}
    for idx, product_id in enumerate(PRODUCT_CATALOG['ProductID']):
        if str(product_id) in purchased_ids:
            combined_scores[idx] *= 0.3
    
    # Get top candidates
    candidate_pool_size = min(200, valid_mask.sum())
    candidate_indices = np.argsort(combined_scores)[-candidate_pool_size:][::-1]
    
    # ML scoring
    candidate_features = PRODUCT_FEATURES[candidate_indices]
    ml_scores = RECOMMENDER_MODEL.predict_proba(candidate_features)[:, 1]
    
    # Final selection
    top_indices = np.argsort(ml_scores)[-top_k:][::-1]
    final_indices = candidate_indices[top_indices]
    
    # Format results
    recommendations = []
    for idx in final_indices:
        product = PRODUCT_CATALOG.iloc[idx]
        is_repurchase = str(product['ProductID']) in purchased_ids
        
        recommendations.append({
            'product_id': str(product['ProductID']),
            'product_name': product['ProductName'],
            'brand': product['ProductBrand'],
            'price': float(product['Price (INR)']),
            'color': product['PrimaryColor'],
            'gender': product['Gender'],
            'category': product.get('Category', 'Unknown'),
            'description': product['Description'][:150] + '...',
            'is_repurchase': is_repurchase
        })
    
    print(f" Returning {len(recommendations)} recommendations")
    return recommendations

def recommendation_agent(state: State) -> State:
    """ML decides products, LLM explains them (NO hallucination)"""

    print(f"\n{'='*80}")
    print(f" RECOMMENDATION AGENT EXECUTING")
    print(f"{'='*80}")

    query = state.get('task', '')
    customer_context = state.get('customer_context') or {}
    parsed_filters = state.get('parsed_filters', {})

    recommendation_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # ───────────────────────────────────────────────
    # Guest clarification
    # ───────────────────────────────────────────────
    has_profile = customer_context.get('profile') is not None
    has_filters = len(parsed_filters) > 0

    if not has_profile and not has_filters and not parsed_filters.get('category') and not parsed_filters.get('max_price'):
        clarification = (
            "I’d love to help you find the right product \n\n"
            "Could you please tell me:\n"
            "• What type of product? (shoes, shirts, pants, etc.)\n"
            "• Any preferred brands?\n"
            "• Your budget range?"
        )

        return {
            'messages': [AIMessage(content=clarification, name="recommendation_agent")],
            'results': clarification,
            'recommendations': clarification,
            'needs_clarification': True,
            'current_agent': 'Recommendation Agent',
            'agents_executed': ['Recommendation Agent']
        }

    # ───────────────────────────────────────────────
    # ML filters
    # ───────────────────────────────────────────────
    ml_filters = {}

    if parsed_filters.get('gender'):
        ml_filters['gender'] = parsed_filters['gender']
    elif customer_context.get('profile') and customer_context['profile'].get('gender'):
        ml_filters['gender'] = customer_context['profile']['gender']

    if parsed_filters.get('brand'):
        ml_filters['brand'] = parsed_filters['brand']

    if parsed_filters.get('max_price'):
        ml_filters['max_price'] = parsed_filters['max_price']

    if parsed_filters.get('min_price'):
        ml_filters['min_price'] = parsed_filters['min_price']

    if parsed_filters.get('category'):
        ml_filters['category'] = parsed_filters['category']
    elif parsed_filters.get('product_type'):
        type_to_category = {
            'shoes': 'Shoes',
            'sneakers': 'Shoes',
            'boots': 'Shoes',
            'shirt': 'Topwear',
            't-shirt': 'Topwear',
            'pants': 'Bottomwear',
            'jeans': 'Bottomwear',
            'trousers': 'Bottomwear',
            'dress': 'Dress',
            'jacket': 'Topwear'
        }
        pt = parsed_filters['product_type'].lower()
        ml_filters['category'] = type_to_category.get(pt, pt.title())

    print(" ML Filters:", ml_filters)

    # ───────────────────────────────────────────────
    # ML decides products (SOURCE OF TRUTH)
    # ───────────────────────────────────────────────
    recommendations = get_ml_recommendations(
        query=query,
        customer_context=customer_context,
        filters=ml_filters,
        use_history=parsed_filters.get('use_purchase_history', False),
        top_k=5
    )

    # ───────────────────────────────────────────────
    # No stock
    # ───────────────────────────────────────────────
    if not recommendations:
        msg = "Sorry, we currently don’t have products matching your exact request. Would you like me to try a different brand, budget or category?"
        return {
            'messages': [AIMessage(content=msg, name="recommendation_agent")],
            'results': msg,
            'recommendations': msg,
            'current_agent': 'Recommendation Agent',
            'agents_executed': ['Recommendation Agent']
        }

    # ───────────────────────────────────────────────
    # LLM PRESENTATION (SAFE MODE)
    # ───────────────────────────────────────────────
    product_json = json.dumps(recommendations, indent=2)

    customer_profile = customer_context.get('profile', {})
    customer_name = customer_profile.get("name", "")

    llm_prompt = f"""
You are a shopping assistant.

STRICT RULES:
- You MUST ONLY use the products listed below
- You MUST copy product_name, price, color, and description exactly
- You may NOT rewrite or change any product data
- Do NOT output JSON
- Do NOT show field names like "product_name" or "price"
- Present products as a clean numbered shopping list

Customer name: {customer_name}
Customer request: {query}

REAL INVENTORY:
{product_json}

FORMAT REQUIRED:

Start with a short friendly sentence.

Then for each product:

1. Product Name by Brand
   Price: INR ...
   Color: ...
   Description: ...

After listing all products, add one short sentence explaining why these match the request.
"""



    final_response = recommendation_llm.invoke(
        [HumanMessage(content=llm_prompt)]
    ).content

    return {
        'messages': [AIMessage(content=final_response, name="recommendation_agent")],
        'results': final_response,
        'recommendations': final_response,
        'current_agent': 'Recommendation Agent',
        'agents_executed': ['Recommendation Agent'],
        'needs_clarification': False
    }



import uuid
from datetime import datetime

# IN-MEMORY ESCALATION STORAGE

ESCALATIONS = []


# ESCALATION AGENT

def escalation_agent(state: State) -> State:
    """Evaluates escalation and sends to supervisor dashboard"""
    
    print(f"\n{'='*80}")
    print(f"  ESCALATION AGENT EXECUTING")
    print(f"{'='*80}")
    
    escalation_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    query = state.get('task', '')
    previous_results = state.get('results', '')
    customer_context = state.get('customer_context', {})
    
    customer_value = ""
    if customer_context.get('insights'):
        insights = customer_context['insights']
        customer_value = f"\n**Customer Value:** {insights['total_purchases']} purchases, ₹{insights['total_spent']:.2f} spent"
    
    prompt = f"""Evaluate if this needs human escalation.

**Query:** {query}
important instructions 
do not use * # or anything else in the query just give plain text as instructed 
**Previous Work:** {previous_results}
{customer_value}

**Escalate (0.7-1.0) if:**
- Billing/payment issues
- Customer angry/frustrated
- High-value customer complaint
- Complex exceptions
- Duplicate charges or refund issues

**Don't Escalate (0.0-0.3) if:**
- Simple query answered well
- General information request

Provide score and reasoning."""
    
    escalation_llm_with_schema = escalation_llm.with_structured_output(Escalation_Schema)
    
    try:
        response = escalation_llm_with_schema.invoke([HumanMessage(content=prompt)])
        confidence = float(response.confidence_score)
        reasoning = response.reasoning
        print(f" Escalation Score: {confidence:.2f}")
        print(f" Reasoning: {reasoning}")
    except Exception as e:
        print(f" Error: {e}")
        confidence = 0.5
        reasoning = "Error in analysis"
    
    if confidence >= 0.7:
        print(" ESCALATING TO SUPERVISOR DASHBOARD")
        
        # Create escalation record
        profile = customer_context.get('profile', {})
        insights = customer_context.get('insights', {})
        purchase_history = customer_context.get('purchase_history', [])
        
        escalation_record = {
            'escalation_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'query': query,
            'confidence_score': confidence,
            'reasoning': reasoning,
            'customer_id': profile.get('customer_id', 'UNKNOWN'),
            'customer_name': profile.get('name', 'Unknown'),
            'customer_email': profile.get('email', 'N/A'),
            'customer_phone': profile.get('phone', 'N/A'),
            'customer_gender': profile.get('gender', 'N/A'),
            'total_purchases': insights.get('total_purchases', 0),
            'total_spent': insights.get('total_spent', 0.0),
            'avg_purchase_value': insights.get('avg_purchase_value', 0.0),
            'favorite_brands': insights.get('favorite_brands', []),
            'favorite_categories': insights.get('favorite_categories', []),
            'purchase_history': purchase_history[:10],
            'resolved_at': None,
            'supervisor_response': None
        }
        
        ESCALATIONS.append(escalation_record)
        print(f" Escalation record created: {escalation_record['escalation_id']}")
        
        customer_name = profile.get('name', '')
        greeting = f"{customer_name}, " if customer_name else ""
        
        # ... inside if confidence >= 0.7: block ...
        
        user_message = f"""Dear {greeting}

I completely understand your frustration, and I sincerely apologize for the inconvenience you are experiencing.

I have immediately escalated your concern to our senior support team. Your case has been flagged as high priority.

What happens next:
- A senior specialist will review your case immediately.
- They have full access to your account details and purchase history.
- You will be contacted within the next 2 hours.
- We will ensure this is resolved to your complete satisfaction.

Your patience is greatly appreciated. We truly value your business and will make this right.

Is there anything else I can help you with in the meantime?"""
        
        escalated = [query]
        
    else:
        user_message = f"Thank you for reaching out. Your query has been processed successfully."
        escalated = []
        print(" No escalation needed")
    
    return {
        'messages': [AIMessage(content=user_message, name="escalation_agent")],
        'confidence_score': confidence,
        'results': user_message,
        'current_agent': 'Escalation Agent',
        'agents_executed': ['Escalation Agent'],
        'escalated_queries': escalated,
        'needs_clarification': False
    }


# ============================================================================
# SUPERVISOR DASHBOARD ROUTES
# ============================================================================

@app.route('/escalations', methods=['GET'])
def get_escalations():
    """Get all escalations for supervisor dashboard"""
    try:
        sorted_escalations = sorted(
            ESCALATIONS, 
            key=lambda x: x['timestamp'], 
            reverse=True
        )
        
        return jsonify({
            'escalations': sorted_escalations,
            'total': len(ESCALATIONS),
            'pending': len([e for e in ESCALATIONS if e['status'] == 'pending']),
            'resolved': len([e for e in ESCALATIONS if e['status'] == 'resolved'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/escalations/<escalation_id>/resolve', methods=['POST'])
def resolve_escalation(escalation_id: str):
    """Mark an escalation as resolved"""
    try:
        data = request.get_json()
        supervisor_response = data.get('response', '')
        
        escalation = next((e for e in ESCALATIONS if e['escalation_id'] == escalation_id), None)
        
        if not escalation:
            return jsonify({'error': 'Escalation not found'}), 404
        
        escalation['status'] = 'resolved'
        escalation['resolved_at'] = datetime.now().isoformat()
        escalation['supervisor_response'] = supervisor_response
        
        print(f"Escalation {escalation_id} marked as resolved")
        
        return jsonify({
            'message': 'Escalation resolved successfully',
            'escalation': escalation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/escalations/<escalation_id>', methods=['GET'])
def get_escalation_details(escalation_id: str):
    """Get details of a specific escalation"""
    try:
        escalation = next((e for e in ESCALATIONS if e['escalation_id'] == escalation_id), None)
        
        if not escalation:
            return jsonify({'error': 'Escalation not found'}), 404
        
        return jsonify(escalation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/escalations/stats', methods=['GET'])
def get_escalation_stats():
    """Get escalation statistics"""
    try:
        total = len(ESCALATIONS)
        pending = len([e for e in ESCALATIONS if e['status'] == 'pending'])
        resolved = len([e for e in ESCALATIONS if e['status'] == 'resolved'])
        
        resolved_escalations = [e for e in ESCALATIONS if e['status'] == 'resolved' and e['resolved_at']]
        
        avg_resolution_time = None
        if resolved_escalations:
            total_time = 0
            for esc in resolved_escalations:
                created = datetime.fromisoformat(esc['timestamp'])
                resolved = datetime.fromisoformat(esc['resolved_at'])
                total_time += (resolved - created).total_seconds()
            avg_resolution_time = total_time / len(resolved_escalations) / 3600
        
        return jsonify({
            'total_escalations': total,
            'pending': pending,
            'resolved': resolved,
            'avg_resolution_hours': round(avg_resolution_time, 2) if avg_resolution_time else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def route_from_orchestrator(state: State) -> str:
    """Routes based on orchestrator decision"""
    next_agent = state.get('next_agent', 'Finish')
    print(f" ROUTER: '{next_agent}'")
    return next_agent


def build_customer_support_graph():
    """Build the complete agent graph"""
    
    graph = StateGraph(State)
    
    graph.add_node('Orchestrator', orchestrator)
    graph.add_node('Query Parser', query_parser_agent)
    graph.add_node('Support Agent', support_agent)
    graph.add_node('Recommendation Agent', recommendation_agent)
    graph.add_node('Escalation Agent', escalation_agent)
    
    graph.set_entry_point('Orchestrator')
    
    graph.add_conditional_edges(
        'Orchestrator',
        route_from_orchestrator,
        {
            "Query Parser": "Query Parser",
            "Support Agent": "Support Agent",
            "Recommendation Agent": "Recommendation Agent",
            "Escalation Agent": "Escalation Agent",
            "Finish": END
        }
    )
    
    graph.add_edge("Query Parser", "Orchestrator")
    graph.add_edge("Support Agent", "Orchestrator")
    graph.add_edge("Recommendation Agent", "Orchestrator")
    graph.add_edge("Escalation Agent", "Orchestrator")
    
    return graph.compile()


def run_customer_support_streaming(customer_query: str, customer_id: str) -> Generator[Dict, None, None]:
    """Run customer support with streaming and database"""
    
    customer_context = get_customer_context(customer_id)
    
    customer_db.add_message(customer_id, 'user', customer_query)
    
    initial_state = State(
        messages=[HumanMessage(content=customer_query)],
        task=customer_query,
        next_agent="",
        current_agent="",
        iteration=0,
        recommendations="",
        results="",
        confidence_score=0.0,
        escalated_queries=[],
        agents_executed=[],
        needs_clarification=False,
        clarification_question="",
        clarification_context="",
        customer_id=customer_id,
        customer_context=customer_context,
        parsed_filters={}
    )
    
    try:
        workflow = build_customer_support_graph()
        
        final_state = workflow.invoke(initial_state)
        
        current_agent = final_state.get('current_agent', 'Unknown')
        agents_executed = final_state.get('agents_executed', [])
        
        for agent_name in agents_executed:
            if agent_name != current_agent:
                agent_output = f"[{agent_name}] Processing..."
                customer_db.add_message(
                    customer_id,
                    'system',
                    agent_output,
                    agent_name=agent_name,
                    metadata={'iteration': 'intermediate'}
                )
        
        yield {'type': 'agent_start', 'agent': current_agent}
        
        response_text = final_state.get('results', '')
        
        customer_db.add_message(
            customer_id, 
            'assistant', 
            response_text,
            agent_name=current_agent,
            metadata={
                'confidence_score': final_state.get('confidence_score', 0.0),
                'needs_clarification': final_state.get('needs_clarification', False),
                'parsed_filters': final_state.get('parsed_filters', {}),
                'agents_executed': agents_executed
            }
        )
        
        yield {'type': 'token', 'content': response_text}

        
        yield {'type': 'agent_complete', 'agent': current_agent}
        
        metadata = {
            'agent': current_agent,
            'needs_clarification': final_state.get('needs_clarification', False),
            'confidence_score': final_state.get('confidence_score', 0.0),
            'escalated': len(final_state.get('escalated_queries', [])) > 0,
            'parsed_filters': final_state.get('parsed_filters', {}),
            'all_agents': agents_executed
        }
        
        yield {'type': 'done', 'metadata': metadata}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"Error: {str(e)}"
        customer_db.add_message(
            customer_id,
            'system',
            error_msg,
            agent_name='System',
            metadata={'error': True}
        )
        
        yield {'type': 'error', 'message': str(e)}


# FLASK ROUTES

@app.route('/')
def serve_frontend():
    return app.send_static_file('index.html')
def create_sse_message(event_type: str, data: dict) -> str:
    """Create SSE message"""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "message": "Customer Support AI is running",
        "ml_models_loaded": ML_LOADED,
        "database_connected": True
    })


# 6. UPDATE LOGIN TO RETURN SESSION TOKEN
@app.route("/login", methods=["POST"])
def login():
    """Login endpoint with session management"""
    try:
        data = request.get_json()
        email = data.get("email")

        if not email:
            return jsonify({"error": "Email required"}), 400

        customer_id = customer_db.get_customer_by_email(email)
        if not customer_id:
            return jsonify({"error": "Email not registered"}), 401

        context = get_customer_context(customer_id)
        
        # Get conversation history
        conversation_history = customer_db.get_conversation_history(customer_id, limit=20)

        return jsonify({
            "success": True,
            "customer_id": customer_id,
            "email": email,
            "profile": context["profile"],
            "insights": context["insights"],
            "conversation_history": conversation_history,
            "session_created": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Main chat endpoint with streaming and session persistence"""

    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        email = data.get('email')
        customer_id = data.get('customer_id')

        # Determine customer_id
        if email:
            customer_id = customer_db.get_customer_by_email(email)
            if not customer_id:
                return jsonify({
                    "error": "Session expired. Please login again.",
                    "code": "SESSION_EXPIRED"
                }), 401
        elif not customer_id:
            customer_id = "GUEST"

        if not user_message:
            return jsonify({'error': 'Message is required'}), 400

        response_headers = {
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'X-Customer-ID': customer_id,
            'X-Customer-Email': email if email else 'GUEST'
        }

        def generate():
            try:
                yield create_sse_message('session_info', {
                    'customer_id': customer_id,
                    'email': email if email else None
                })
                
                yield create_sse_message('start', {'message': 'Processing...'})

                for event in run_customer_support_streaming(user_message, customer_id):
                    if event['type'] == 'agent_start':
                        yield create_sse_message('agent_start', {'agent': event['agent']})
                    elif event['type'] == 'token':
                        yield create_sse_message('token', {'content': event['content']})
                    elif event['type'] == 'agent_complete':
                        yield create_sse_message('agent_complete', {'agent': event['agent']})
                    elif event['type'] == 'done':
                        yield create_sse_message('done', {'metadata': event.get('metadata', {})})
                    elif event['type'] == 'error':
                        yield create_sse_message('error', {'message': event['message']})
                        break

            except Exception as e:
                print(f" Generate error: {e}")
                import traceback
                traceback.print_exc()
                yield create_sse_message('error', {
                    'message': f'Error: {str(e)}'
        })

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers=response_headers
        )

    except Exception as e:
        print(f" Chat error: {e}")
        import traceback
        traceback.print_exc()
        
        def error_stream():
            yield create_sse_message('error', {
                'message': str(e)
            })
        
        return Response(error_stream(), mimetype='text/event-stream')


# 7. ADD LOGOUT ENDPOINT
@app.route('/logout', methods=['POST'])
def logout():
    """Logout endpoint"""
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        # Optional: Add any cleanup logic here
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/customer/<customer_id>', methods=['GET'])
def get_customer_profile(customer_id: str):
    """Get customer profile and history"""
    try:
        context = get_customer_context(customer_id)
        if not context.get('profile'):
            return jsonify({'error': 'Customer not found'}), 404
        
        return jsonify({
            'profile': context['profile'],
            'purchase_history': context['purchase_history'],
            'insights': context['insights']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/customer', methods=['POST'])
def create_customer():
    """Create new customer"""
    try:
        data = request.get_json()
        success = customer_db.add_customer(
            customer_id=data.get('customer_id'),
            name=data.get('name'),
            email=data.get('email'),
            phone=data.get('phone'),
            gender=data.get('gender')
        )
        
        if success:
            return jsonify({'message': 'Customer created successfully'})
        else:
            return jsonify({'error': 'Customer already exists'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/purchase', methods=['POST'])
def add_purchase():
    """Add purchase record"""
    try:
        data = request.get_json()
        customer_db.add_purchase(
            customer_id=data.get('customer_id'),
            product_id=data.get('product_id'),
            product_name=data.get('product_name'),
            product_brand=data.get('product_brand'),
            category=data.get('category'),
            price=data.get('price'),
            rating=data.get('rating'),
            purchase_date=data.get('purchase_date')
        )
        return jsonify({'message': 'Purchase recorded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# 5. UPDATE CONVERSATION ENDPOINT TO HANDLE MISSING SESSIONS
@app.route('/conversation/<customer_id>', methods=['GET'])
def get_conversation(customer_id: str):
    """Get conversation history - with session validation"""
    try:
        # Verify customer exists
        profile = customer_db.get_customer(customer_id)
        if not profile and customer_id != "GUEST":
            return jsonify({
                'error': 'Customer not found',
                'conversation_history': [],
                'should_logout': True
            }), 404
        
        limit = request.args.get('limit', 20, type=int)
        history = customer_db.get_conversation_history(customer_id, limit=limit)
        
        return jsonify({
            'conversation_history': history,
            'customer_id': customer_id,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting conversation: {e}")
        return jsonify({
            'error': str(e),
            'conversation_history': []
        }), 500

# 4. ADD KEEP-ALIVE ENDPOINT
@app.route('/session/ping', methods=['POST'])
def session_ping():
    """Keep session alive"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'alive': True, 'guest': True})
        
        customer_id = customer_db.get_customer_by_email(email)
        if not customer_id:
            return jsonify({
                'alive': False,
                'error': 'Session expired'
            }), 401
        
        return jsonify({
            'alive': True,
            'customer_id': customer_id,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'alive': False, 'error': str(e)}), 500

# 3. ADD SESSION REFRESH ENDPOINT
@app.route('/session/refresh', methods=['POST'])
def refresh_session():
    """Refresh user session and return current state"""
    try:
        data = request.get_json()
        email = data.get('email')
        customer_id = data.get('customer_id')
        
        if email:
            customer_id = customer_db.get_customer_by_email(email)
            if not customer_id:
                return jsonify({
                    'error': 'Session expired',
                    'valid': False
                }), 401
        elif not customer_id:
            return jsonify({
                'error': 'No session found',
                'valid': False
            }), 400
        
        # Get fresh context
        context = get_customer_context(customer_id)
        
        return jsonify({
            'valid': True,
            'customer_id': customer_id,
            'profile': context['profile'],
            'insights': context['insights'],
            'message': 'Session refreshed'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'valid': False
        }), 500



@app.route('/reset/<customer_id>', methods=['POST'])
def reset_conversation(customer_id: str):
    """Reset conversation history"""
    try:
        customer_db.clear_conversation_history(customer_id)
        return jsonify({'message': 'Conversation reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 1. ADD SESSION VALIDATION MIDDLEWARE (Add after CORS setup)
from functools import wraps

def validate_session(f):
    """Middleware to validate customer session"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # For GET requests, check query params
            if request.method == 'GET':
                customer_id = request.args.get('customer_id')
                email = request.args.get('email')
            # For POST requests, check JSON body
            else:
                data = request.get_json()
                customer_id = data.get('customer_id') if data else None
                email = data.get('email') if data else None
            
            # If neither provided, allow (for guest users)
            if not customer_id and not email:
                return f(*args, **kwargs)
            
            # If email provided, verify it exists
            if email:
                verified_id = customer_db.get_customer_by_email(email)
                if not verified_id:
                    return jsonify({
                        'error': 'Session expired',
                        'code': 'SESSION_EXPIRED'
                    }), 401
            
            return f(*args, **kwargs)
        except Exception as e:
            print(f"Session validation error: {e}")
            return f(*args, **kwargs)  # Allow to continue on validation errors
    
    return decorated_function




@app.route('/debug/ml-status', methods=['GET'])
def ml_status():
    """Debug endpoint to check ML model status"""
    try:
        status = {
            'ml_loaded': ML_LOADED,
            'models': {
                'recommender_model': RECOMMENDER_MODEL is not None,
                'feature_builder': FEATURE_BUILDER is not None,
                'text_model': TEXT_MODEL is not None,
            },
            'data': {
                'product_catalog_loaded': PRODUCT_CATALOG is not None,
                'product_catalog_size': len(PRODUCT_CATALOG) if PRODUCT_CATALOG is not None else 0,
                'product_features_shape': PRODUCT_FEATURES.shape if PRODUCT_FEATURES is not None else None,
                'product_embeddings_shape': PRODUCT_EMBEDDINGS.shape if PRODUCT_EMBEDDINGS is not None else None,
            },
            'file_paths': {
                'recommender_model': r"path to recommender_model.pkl",
                'feature_builder': r"path to feature_builder.pkl",
                'product_catalog': r"path to myntra_products_catalog.csv",
            },
            'status_message': 'ML models loaded and ready!' if ML_LOADED else 'ML models NOT loaded - NO FALLBACK'
        }
        
        import os
        status['files_exist'] = {
            'recommender_model': os.path.exists(r"path to recommender_model.pkl"),
            'feature_builder': os.path.exists(r"path to feature_builder.pkl"),
            'product_catalog': os.path.exists(r" path to myntra_products_catalog.csv"),
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'ml_loaded': False,
            'status_message': 'Error checking ML status'
        }), 500


@app.route('/debug/test-recommendation', methods=['POST'])
def test_recommendation():
    """Test ML recommendation with sample query"""
    try:
        data = request.get_json()
        test_query = data.get('query', 'show me nike shoes')
        customer_id = data.get('customer_id', 'CUST001')
        
        customer_context = get_customer_context(customer_id)
        
        test_filters = {
            'brand': 'Nike',
            'category': 'Shoes',
            'max_price': 5000
        }
        
        if not ML_LOADED:
            return jsonify({
                'status': 'error',
                'message': 'ML models not loaded - NO FALLBACK AVAILABLE',
                'using_fallback': False
            })
        
        recommendations = get_ml_recommendations(
            query=test_query,
            customer_context=customer_context,
            filters=test_filters,
            use_history=False,
            top_k=3
        )
        
        return jsonify({
            'status': 'success',
            'ml_loaded': True,
            'query': test_query,
            'filters_applied': test_filters,
            'recommendations_count': len(recommendations),
            'recommendations': recommendations,
            'message': f'ML model working! Found {len(recommendations)} products.'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/debug/product-stats', methods=['GET'])
def product_stats():
    """Get product catalog statistics"""
    try:
        if PRODUCT_CATALOG is None:
            return jsonify({'error': 'Product catalog not loaded'}), 500
        
        stats = {
            'total_products': len(PRODUCT_CATALOG),
            'brands': {
                'total': PRODUCT_CATALOG['ProductBrand'].nunique(),
                'top_5': PRODUCT_CATALOG['ProductBrand'].value_counts().head(5).to_dict()
            },
            'categories': {
                'total': PRODUCT_CATALOG['Category'].nunique() if 'Category' in PRODUCT_CATALOG.columns else 0,
                'distribution': PRODUCT_CATALOG['Category'].value_counts().to_dict() if 'Category' in PRODUCT_CATALOG.columns else {}
            },
            'price_range': {
                'min': float(PRODUCT_CATALOG['Price (INR)'].min()),
                'max': float(PRODUCT_CATALOG['Price (INR)'].max()),
                'avg': float(PRODUCT_CATALOG['Price (INR)'].mean())
            },
            'gender_distribution': PRODUCT_CATALOG['Gender'].value_counts().to_dict() if 'Gender' in PRODUCT_CATALOG.columns else {}
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print(" CUSTOMER SUPPORT AI - COMPLETE SYSTEM (NO LLM HALLUCINATION)")
    print("="*80)
    
    print("\n Initializing ML models...")
    initialize_ml_models()
    
    print("\n📡 Server starting on http://localhost:5000")
    print(" Endpoints:")
    print("   - POST /chat - Main chat endpoint")
    print("   - POST /login - User login")
    print("   - GET  /health - Health check")
    print("   - GET  /customer/<id> - Get customer profile")
    print("   - POST /customer - Create customer")
    print("   - POST /purchase - Add purchase")
    print("   - GET  /conversation/<id> - Get chat history")
    print("   - POST /reset/<id> - Reset conversation")
    print("   - GET  /escalations - Get all escalations (SUPERVISOR)")
    print("   - POST /escalations/<id>/resolve - Resolve escalation (SUPERVISOR)")
    
    print("\n🔧 Debug Endpoints:")
    print("   - GET  /debug/ml-status - Check ML model status")
    print("   - POST /debug/test-recommendation - Test ML with sample query")
    print("   - GET  /debug/product-stats - Get product catalog statistics")
    
    print("\n Agent Architecture:")
    print("   1. Orchestrator - Routes queries")
    print("   2. Query Parser - Extracts filters")
    print("   3. Support Agent - Handles FAQs")
    print("   4. Recommendation Agent - Provides products (ML ONLY, NO HALLUCINATION)")
    print("   5. Escalation Agent - Evaluates issues")
    
    if ML_LOADED:
        print("\n ML Models: LOADED AND READY")
        print("    Recommendation Agent: Will ONLY show products from catalog")
        print("     NO LLM fallback - honest out-of-stock messages instead")
    else:
        print("\n  ML Models: NOT LOADED")
        print("    Recommendation Agent will return error message")
        print("    NO hallucinated products will be generated")
    
    print("\nReady!\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )
