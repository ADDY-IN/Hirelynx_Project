from sqlalchemy import create_engine, text
from app.config import settings

def check_candidates_table():
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    try:
        with engine.connect() as conn:
            # Check schema of candidates table
            print("--- Candidates Table Schema ---")
            res = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'candidates'"))
            for row in res:
                print(row)
            
            # Check count
            res = conn.execute(text("SELECT count(*) FROM candidates"))
            count = res.fetchone()[0]
            print(f"\nTotal candidates in 'candidates' table: {count}")
            
            # Check first candidate's personal_details
            print("\n--- First Candidate Personal Details ---")
            res = conn.execute(text("SELECT personal_details FROM candidates LIMIT 1"))
            row = res.fetchone()
            if row:
                print(row[0])
            
            # Check first 5 candidates
            print("\n--- First 5 Candidates ---")
            res = conn.execute(text("SELECT * FROM candidates LIMIT 5"))
            for row in res:
                print(row)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    check_candidates_table()
