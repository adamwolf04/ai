import sqlite3
import json

conn = sqlite3.connect('logs/lineage.db')
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT status, messages FROM run_traces WHERE status != 'success' LIMIT 1").fetchone()

if row:
    print(f"Status: {row['status']}")
    messages = json.loads(row['messages'])
    if messages:
        print(f"Messages: {json.dumps(messages[-1], indent=2)}")
    else:
        print("No messages")
else:
    print("No non-success runs found")

# Also check for actual error strings
row = conn.execute("SELECT cause_code FROM evaluations WHERE cause_code IS NOT NULL AND cause_code != 'OK' LIMIT 1").fetchone()
if row:
    print(f"Error: {row['cause_code']}")
