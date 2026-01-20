

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from typing import Dict
import os
from datetime import datetime
import json
import sys
sys.path.append(r"path to database.py")
from Database.database import CustomerDatabase
from functools import wraps

# Import agent system
from agent import (
    initialize_ml_models,
    run_customer_support_streaming,
    ML_LOADED,
    ESCALATIONS,
    PRODUCT_CATALOG,
    get_ml_recommendations
)

# FLASK APP SETUP
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)

# DATABASE INITIALIZATION
customer_db = CustomerDatabase()
print("Using DB:", os.path.abspath(customer_db.db_path))


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


def create_sse_message(event_type: str, data: dict) -> str:
    """Create SSE message"""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


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


# FLASK ROUTES

@app.route('/')
def serve_frontend():
    return app.send_static_file('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "message": "Customer Support AI is running",
        "ml_models_loaded": ML_LOADED,
        "database_connected": True
    })


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

                customer_context = get_customer_context(customer_id)

                for event in run_customer_support_streaming(user_message, customer_id, customer_context, customer_db):
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


# ESCALATION ENDPOINTS

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


# DEBUG ENDPOINTS

@app.route('/debug/ml-status', methods=['GET'])
def ml_status():
    """Debug endpoint to check ML model status"""
    try:
        from agent import RECOMMENDER_MODEL, FEATURE_BUILDER, TEXT_MODEL, PRODUCT_FEATURES, PRODUCT_EMBEDDINGS
        
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


# ERROR HANDLERS

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
    
    print("\n Server starting on http://localhost:5000")
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
    
    print("\n Debug Endpoints:")
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
        print("Recommendation Agent: Will ONLY show products from catalog")
    else:
        print("\n  ML Models: NOT LOADED")
        print(" Recommendation Agent will return error message")
    
    print("\nReady!\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )
