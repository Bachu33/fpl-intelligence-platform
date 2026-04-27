import pandas as pd
import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("FATAL: SUPABASE_URL or SUPABASE_KEY not found in environment.")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase client connected.")

def push_to_supabase(df):
    records = df.to_dict(orient="records")
    
    print(f"Pushing {len(records)} records to Supabase...")
    
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("player_gameweek_stats").upsert(batch).execute()
        print(f"  Uploaded records {i+1} to {min(i+batch_size, len(records))}")
    
    print(f"[{datetime.now()}] All records pushed successfully.")

if __name__ == "__main__":
    print("Starting Supabase update...")
    
    df = pd.read_parquet("data/processed/players.parquet")
    print(f"Loaded {len(df)} records from parquet")
    
    push_to_supabase(df)
    
    print("Supabase update complete.")
