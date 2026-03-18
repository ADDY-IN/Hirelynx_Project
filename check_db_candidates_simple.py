from sqlalchemy import create_engine, text
from app.config import settings

def check_candidates_count():
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT count(*) FROM candidates"))
            count = res.fetchone()[0]
            print(f"TOTAL_CANDIDATES_COUNT:{count}")
            
            # Also check for Sonia Sharma specifically
            res = conn.execute(text("SELECT count(*) FROM candidates WHERE resume_parsed_json::text ILIKE '%Sonia%'"))
            sonia_count = res.fetchone()[0]
            print(f"SONIA_COUNT:{sonia_count}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    check_candidates_count()
