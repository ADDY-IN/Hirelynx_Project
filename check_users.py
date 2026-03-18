from sqlalchemy import create_engine, text
from app.config import settings

def check_users():
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    try:
        with engine.connect() as conn:
            print("--- Searching 'users' table ---")
            # First check schema of users table
            res = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"))
            cols = [r[0] for r in res]
            print(f"Columns: {cols}")
            
            # Simple broad search in users
            query = "SELECT * FROM users WHERE \"firstName\" ILIKE '%Sonia%' OR \"lastName\" ILIKE '%Sonia%' OR \"email\" ILIKE '%Sonia%'"
            if "firstName" not in cols:
                # Try generic text search if columns don't match
                query = "SELECT * FROM users LIMIT 10"
                
            res = conn.execute(text(query))
            rows = res.fetchall()
            print(f"User search results: {len(rows)}")
            for r in rows[:5]:
                print(r)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    check_users()
