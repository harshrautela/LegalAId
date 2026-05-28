import sqlite3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DB_DIR = ROOT / "database"
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "legalaid.db"

PROCESSED = ROOT / "data" / "processed"

FILES = [
    ("summarisation_train.jsonl","summarisation"),
    ("qa_train.jsonl","qa"),
    ("rag_train.jsonl","rag")
]


def load_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


conn=sqlite3.connect(DB_PATH)
cur=conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS documents(
id INTEGER PRIMARY KEY AUTOINCREMENT,
task TEXT,
doc_id TEXT,
content TEXT,
metadata TEXT
)
""")

inserted=0

for file_name,task in FILES:

    path=PROCESSED/file_name

    rows=load_jsonl(path)

    for r in rows:

        content=(
            r.get("text")
            or r.get("context")
            or ""
        )

        doc_id=r.get("doc_id","")

        cur.execute(
        """
        INSERT INTO documents(task,doc_id,content,metadata)
        VALUES(?,?,?,?)
        """,
        (
          task,
          doc_id,
          content,
          json.dumps(r)
        )
        )

        inserted+=1

conn.commit()
conn.close()

print(f"Inserted {inserted} rows")
print(f"SQLite DB created at:\n{DB_PATH}")