from sqlalchemy import create_engine, text
from app.config import settings

def check_all_names():
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    try:
        with engine.connect() as conn:
            print("--- Names in candidate_profiles ---")
            res = conn.execute(text("SELECT id, \"personalDetails\" FROM candidate_profiles LIMIT 10"))
            for row in res:
                print(row)
            
            print("\n--- Names in candidates ---")
            res = conn.execute(text("SELECT id, resume_parsed_json FROM candidates"))
            for row in res:
                # Extract snippet if available
                snippet = ""
                if row[1] and isinstance(row[1], dict):
                    snippet = row[1].get('text', '')[:100]
                print(f"ID: {row[0]}, Resume: {snippet}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    check_all_names()
