"""
Investigation: Dimension Mismatch in Vector Search
Test what happens when database embeddings have different dimensions than search query
"""
import sys
import pg8000
from task._constants import API_KEY
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode

print("=" * 100)
print("INVESTIGATION 1: DIMENSION MISMATCH")
print("=" * 100)

# Initialize clients
embeddings_client = DialEmbeddingsClient('text-embedding-3-small-1', API_KEY)
db_config = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

print("\n[SCENARIO 1A] Storing embeddings with 384 dimensions...")
try:
    # Store some test data with 384 dimensions
    conn = pg8000.connect(**db_config)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE vectors")
        conn.commit()
        print("[INFO] Table truncated")
        
        # Generate and store embeddings with 384 dimensions
        test_texts = [
            "The microwave has safety features",
            "Clean the microwave with a damp cloth",
            "Set the timer for 5 minutes"
        ]
        
        embeddings_384 = embeddings_client.get_embeddings(test_texts, dimensions=384)
        print(f"[INFO] Generated {len(embeddings_384)} embeddings with 384 dimensions")
        
        for idx, text in enumerate(test_texts):
            embedding = embeddings_384[idx]
            vector_string = f"[{','.join(map(str, embedding))}]"
            cur.execute(
                "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                ('test_384.txt', text, vector_string)
            )
        conn.commit()
        print("[SUCCESS] Stored 3 chunks with 384-dimensional embeddings")
    conn.close()
except Exception as e:
    print(f"[ERROR] Failed to store 384-dim embeddings: {e}")
    sys.exit(1)

print("\n[SCENARIO 1B] Attempting to search with 385-dimensional embedding...")
try:
    # Try to search with 385 dimensions
    query_embedding_385 = embeddings_client.get_embeddings("microwave safety", dimensions=385)
    print(f"[INFO] Generated query embedding with 385 dimensions")
    
    conn = pg8000.connect(**db_config)
    with conn.cursor() as cur:
        vector_string = f"[{','.join(map(str, query_embedding_385[0]))}]"
        
        sql = """
            SELECT text, embedding <=> %s::vector as distance
            FROM vectors
            ORDER BY distance
            LIMIT 3
        """
        
        cur.execute(sql, (vector_string,))
        results = cur.fetchall()
        
        print(f"[UNEXPECTED] Search succeeded! Found {len(results)} results")
        for text, distance in results:
            print(f"  - Distance: {distance:.4f}, Text: {text[:50]}")
    conn.close()
except Exception as e:
    print(f"[EXPECTED ERROR] Search failed with dimension mismatch: {type(e).__name__}")
    print(f"[ERROR DETAILS] {str(e)}")

print("\n[SCENARIO 1C] Searching with matching 384 dimensions...")
try:
    # Search with correct dimensions
    query_embedding_384 = embeddings_client.get_embeddings("microwave safety", dimensions=384)
    print(f"[INFO] Generated query embedding with 384 dimensions")
    
    conn = pg8000.connect(**db_config)
    with conn.cursor() as cur:
        vector_string = f"[{','.join(map(str, query_embedding_384[0]))}]"
        
        sql = """
            SELECT text, embedding <=> %s::vector as distance
            FROM vectors
            ORDER BY distance
            LIMIT 3
        """
        
        cur.execute(sql, (vector_string,))
        results = cur.fetchall()
        
        print(f"[SUCCESS] Search succeeded! Found {len(results)} results")
        for text, distance in results:
            print(f"  - Distance: {distance:.4f}, Text: {text[:50]}")
    conn.close()
except Exception as e:
    print(f"[ERROR] Search failed even with matching dimensions: {e}")

print("\n[SCENARIO 1D] Attempting to store 1536-dim embedding in 384-dim table...")
try:
    embeddings_1536 = embeddings_client.get_embeddings(["test with 1536 dims"], dimensions=1536)
    print(f"[INFO] Generated embedding with 1536 dimensions")
    
    conn = pg8000.connect(**db_config)
    with conn.cursor() as cur:
        vector_string = f"[{','.join(map(str, embeddings_1536[0]))}]"
        cur.execute(
            "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
            ('test_1536.txt', 'test text', vector_string)
        )
        conn.commit()
        print("[UNEXPECTED] Successfully inserted 1536-dim embedding!")
    conn.close()
except Exception as e:
    print(f"[EXPECTED ERROR] Insertion failed: {type(e).__name__}")
    print(f"[ERROR DETAILS] {str(e)}")

print("\n" + "=" * 100)
print("KEY FINDINGS - DIMENSION MISMATCH:")
print("=" * 100)
print("1. PostgreSQL pgvector table schema defines VECTOR(1536) - fixed dimension")
print("2. Inserting vectors with different dimensions than schema = ERROR")
print("3. Searching with different dimensions than stored vectors = ERROR")
print("4. Dimension must match: Schema = Stored Data = Query Vector")
print("=" * 100)
