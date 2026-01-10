import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os

BASE_DIR = r"C:/Users/GS Adithya Krishna\Desktop/internship\backend\Database"
DB_PATH = os.path.join(BASE_DIR, "customer_support.db")

class CustomerDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize all database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Customer Profile Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                gender TEXT,
                preferred_categories TEXT,
                preferred_brands TEXT,
                budget_range TEXT,
                date_of_purchase TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Purchase History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS purchase_history (
                purchase_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                product_brand TEXT,
                category TEXT,
                price REAL NOT NULL,
                purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rating INTEGER,
                review TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        ''')
        
        # Conversation History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                message_role TEXT NOT NULL,
                message_content TEXT NOT NULL,
                agent_name TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        ''')
        
        # Session Table (track active sessions)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    
    # CUSTOMER OPERATIONS 
    
    def add_customer(self, customer_id: str, name: str, email: str, 
                     phone: str = None, gender: str = None) -> bool:
        """Add a new customer"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO customers (customer_id, name, email, phone, gender)
                VALUES (?, ?, ?, ?, ?)
            ''', (customer_id, name, email, phone, gender))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            print(f"Customer {customer_id} already exists")
            return False
        except Exception as e:
            print(f"Error adding customer: {e}")
            return False
    
    def get_customer(self, customer_id: str) -> Optional[Dict]:
        """Get customer profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM customers WHERE customer_id = ?', (customer_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'customer_id': row[0],
                'name': row[1],
                'email': row[2],
                'phone': row[3],
                'gender': row[4],
                'preferred_categories': row[5],
                'preferred_brands': row[6],
                'budget_range': row[7],
                'created_at': row[8],
                'updated_at': row[9]
            }
        return None
    
    def update_customer_preferences(self, customer_id: str, 
                                   preferred_categories: str = None,
                                   preferred_brands: str = None,
                                   budget_range: str = None):
        """Update customer preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if preferred_categories:
            updates.append("preferred_categories = ?")
            params.append(preferred_categories)
        if preferred_brands:
            updates.append("preferred_brands = ?")
            params.append(preferred_brands)
        if budget_range:
            updates.append("budget_range = ?")
            params.append(budget_range)
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(customer_id)
            
            query = f"UPDATE customers SET {', '.join(updates)} WHERE customer_id = ?"
            cursor.execute(query, params)
            conn.commit()
        
        conn.close()
    
    # ==================== PURCHASE HISTORY OPERATIONS ====================
    
    def add_purchase(self, customer_id, product_id, product_name, product_brand,
                category, price, rating, purchase_date):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO purchase_history 
        (customer_id, product_id, product_name, product_brand, category, price, rating, purchase_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (customer_id, product_id, product_name, product_brand,
          category, price, rating, purchase_date))

        conn.commit()
        conn.close()

    
    def get_purchase_history(self, customer_id: str, limit: int = 10) -> List[Dict]:
        """Get customer's purchase history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT product_id, product_name, product_brand, category, 
                   price, purchase_date, rating
            FROM purchase_history
            WHERE customer_id = ?
            ORDER BY purchase_date DESC
            LIMIT ?
        ''', (customer_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'product_id': row[0],
            'product_name': row[1],
            'product_brand': row[2],
            'category': row[3],
            'price': row[4],
            'purchase_date': row[5],
            'rating': row[6]
        } for row in rows]
    
    def get_customer_insights(self, customer_id: str) -> Dict:
        """Get aggregated customer insights"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total purchases and spending
        cursor.execute('''
            SELECT COUNT(*), SUM(price), AVG(price)
            FROM purchase_history
            WHERE customer_id = ?
        ''', (customer_id,))
        
        stats = cursor.fetchone()
        
        # Favorite brands
        cursor.execute('''
            SELECT product_brand, COUNT(*) as count
            FROM purchase_history
            WHERE customer_id = ?
            GROUP BY product_brand
            ORDER BY count DESC
            LIMIT 3
        ''', (customer_id,))
        
        fav_brands = [row[0] for row in cursor.fetchall()]
        
        # Favorite categories
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM purchase_history
            WHERE customer_id = ?
            GROUP BY category
            ORDER BY count DESC
            LIMIT 3
        ''', (customer_id,))
        
        fav_categories = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_purchases': stats[0] or 0,
            'total_spent': stats[1] or 0,
            'avg_purchase_value': stats[2] or 0,
            'favorite_brands': fav_brands,
            'favorite_categories': fav_categories
        }
    
    # CONVERSATION HISTORY OPERATIONS 
    
    def add_message(self, customer_id: str, role: str, content: str, 
                   agent_name: str = None, metadata: Dict = None):
        """Add a message to conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO conversations 
            (customer_id, message_role, message_content, agent_name, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (customer_id, role, content, agent_name, metadata_json))
        
        conn.commit()
        conn.close()
    def get_customer_by_email(self, email: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id FROM customers WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    
    def get_conversation_history(self, customer_id: str, limit: int = 20) -> List[Dict]:
        """Get recent conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT message_role, message_content, agent_name, metadata, timestamp
            FROM conversations
            WHERE customer_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (customer_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Reverse to get chronological order
        history = [{
            'role': row[0],
            'content': row[1],
            'agent': row[2],
            'metadata': json.loads(row[3]) if row[3] else None,
            'timestamp': row[4]
        } for row in reversed(rows)]
        
        return history
    
    
    def clear_conversation_history(self, customer_id: str):
        """Clear conversation history for a customer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM conversations WHERE customer_id = ?', (customer_id,))
        
        conn.commit()
        conn.close()


#  UTILITY FUNCTIONS 

def seed_sample_data(db: CustomerDatabase):
    """Add sample data for testing"""
    
    # Add sample customers
    customers = [
        ('CUST001', 'John Doe', 'john@example.com', '+919876543210', 'Men'),
        ('CUST002', 'Jane Smith', 'jane@example.com', '+919876543211', 'Women'),
        ('CUST003', 'Alex Johnson', 'alex@example.com', '+919876543212', 'Men')
    ]
    
    for cust in customers:
        db.add_customer(*cust)
    now = datetime.now()

    # Add sample purchases
    purchases = [
    ("P1","CUST001","AD001","Adidas Running Shoes","Adidas","Shoes",4999,4.5,(now - timedelta(days=90)).isoformat()),
    ("P2","CUST001","NK001","Nike Training Shoes","Nike","Shoes",5999,4.3,(now - timedelta(days=30)).isoformat()),
    ("P3","CUST001","AD002","Adidas Track Pants","Adidas","Clothing",2999,4.4,(now - timedelta(days=10)).isoformat()),

    ("P4","CUST002","ZR001","Zara Party Dress","Zara","Dress",6999,4.6,(now - timedelta(days=45)).isoformat()),
    ("P5","CUST002","ZR002","Zara Heels","Zara","Shoes",4999,4.2,(now - timedelta(days=15)).isoformat()),
]
    
    for p in purchases:
        db.add_purchase(
        p[1],  # customer_id
        p[2],  # product_id
        p[3],  # product_name
        p[4],  # brand
        p[5],  # category
        p[6],  # price
        p[7],  # rating
        p[8]   # purchase_date
    )

    
    print("Sample data seeded")


if __name__ == '__main__':
    # Initialize database
    db = CustomerDatabase()
    
    # Seed sample data
    seed_sample_data(db)
    
    # Test queries
    print("\n Customer CUST001 Profile:")
    print(db.get_customer('CUST001'))
    
    print("\n Purchase History:")
    print(db.get_purchase_history('CUST001'))
    
    print("\nCustomer Insights:")
    print(db.get_customer_insights('CUST001'))
