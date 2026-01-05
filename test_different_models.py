"""
Investigation: Different Embedding Models for Indexing vs Search
Test what happens when using different models with same dimensions
"""
import sys
import pg8000
from task._constants import API_KEY
from task.embeddings.embeddings_client import DialEmbeddingsClient

print("=" * 100)
print("INVESTIGATION 2: DIFFERENT EMBEDDING MODELS")
print("=" * 100)

db_config = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

# Test if both models are available
print("\n[SETUP] Testing model availability...")
try:
    client_small = DialEmbeddingsClient('text-embedding-3-small-1', API_KEY)
    print("[SUCCESS] text-embedding-3-small-1 is available")
except Exception as e:
    print(f"[ERROR] text-embedding-3-small-1 failed: {e}")
    sys.exit(1)

try:
    client_005 = DialEmbeddingsClient('text-embedding-005', API_KEY)
    test_embedding = client_005.get_embeddings(["test"], dimensions=384)
    print("[SUCCESS] text-embedding-005 is available")
except Exception as e:
    print(f"[WARNING] text-embedding-005 may not be available: {e}")
    print("[INFO] Will test with text-embedding-3-small-1 only")
    client_005 = None

print("\n[SCENARIO 2A] Indexing documents with text-embedding-3-small-1 (384 dims)...")
try:
    # Store documents using text-embedding-3-small-1
    conn = pg8000.connect(**db_config)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE vectors")
        conn.commit()
        print("[INFO] Table truncated")
        
        test_docs = [
            "The microwave oven has multiple power levels",
            "To defrost food, use the defrost button",
            "Never operate the microwave when empty",
            "Clean the interior with mild soap and water",
            "The turntable should rotate during operation"
        ]
        
        embeddings_small = client_small.get_embeddings(test_docs, dimensions=384)
        print(f"[INFO] Generated {len(embeddings_small)} embeddings with text-embedding-3-small-1")
        
        for idx, text in enumerate(test_docs):
            embedding = embeddings_small[idx]
            vector_string = f"[{','.join(map(str, embedding))}]"
            cur.execute(
                "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                ('indexed_with_small.txt', text, vector_string)
            )
        conn.commit()
        print("[SUCCESS] Indexed 5 documents with text-embedding-3-small-1")
    conn.close()
except Exception as e:
    print(f"[ERROR] Indexing failed: {e}")
    sys.exit(1)

print("\n[SCENARIO 2B] Searching with SAME model (text-embedding-3-small-1)...")
try:
    query = "how to clean the microwave"
    query_embedding_small = client_small.get_embeddings([query], dimensions=384)
    
    conn = pg8000.connect(**db_config)
    with conn.cursor() as cur:
        vector_string = f"[{','.join(map(str, query_embedding_small[0]))}]"
        
        sql = """
            SELECT text, embedding <=> %s::vector as distance
            FROM vectors
            ORDER BY distance
            LIMIT 3
        """
        
        cur.execute(sql, (vector_string,))
        results = cur.fetchall()
        
        print(f"[SUCCESS] Found {len(results)} results (SAME MODEL)")
        for i, (text, distance) in enumerate(results, 1):
            similarity = 1.0 - distance
            print(f"  {i}. Similarity: {similarity:.4f} | {text}")
    conn.close()
except Exception as e:
    print(f"[ERROR] Search failed: {e}")

if client_005:
    print("\n[SCENARIO 2C] Searching with DIFFERENT model (text-embedding-005)...")
    try:
        query = "how to clean the microwave"
        query_embedding_005 = client_005.get_embeddings([query], dimensions=384)
        
        conn = pg8000.connect(**db_config)
        with conn.cursor() as cur:
            vector_string = f"[{','.join(map(str, query_embedding_005[0]))}]"
            
            sql = """
                SELECT text, embedding <=> %s::vector as distance
                FROM vectors
                ORDER BY distance
                LIMIT 3
            """
            
            cur.execute(sql, (vector_string,))
            results = cur.fetchall()
            
            print(f"[SUCCESS] Found {len(results)} results (DIFFERENT MODEL)")
            for i, (text, distance) in enumerate(results, 1):
                similarity = 1.0 - distance
                print(f"  {i}. Similarity: {similarity:.4f} | {text}")
        conn.close()
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")

print("\n[SCENARIO 2D] Comparing embedding spaces...")
try:
    # Generate embeddings for same text with both models
    test_text = "microwave oven"
    
    embedding_small = client_small.get_embeddings([test_text], dimensions=384)[0]
    
    if client_005:
        embedding_005 = client_005.get_embeddings([test_text], dimensions=384)[0]
        
        # Calculate cosine similarity between the two embeddings
        import math
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(embedding_small, embedding_005))
        
        # Magnitudes
        mag_small = math.sqrt(sum(x * x for x in embedding_small))
        mag_005 = math.sqrt(sum(x * x for x in embedding_005))
        
        # Cosine similarity
        cosine_sim = dot_product / (mag_small * mag_005)
        
        print(f"[ANALYSIS] Cosine similarity between models for '{test_text}':")
        print(f"  text-embedding-3-small-1 vs text-embedding-005: {cosine_sim:.6f}")
        
        if cosine_sim < 0.8:
            print("  [WARNING] Low similarity indicates different embedding spaces!")
        else:
            print("  [INFO] Relatively high similarity, but spaces may still differ")
    else:
        print("[INFO] Cannot compare - text-embedding-005 not available")
        
except Exception as e:
    print(f"[ERROR] Comparison failed: {e}")

print("\n" + "=" * 100)
print("KEY FINDINGS - DIFFERENT EMBEDDING MODELS:")
print("=" * 100)
print("1. Different models create embeddings in DIFFERENT VECTOR SPACES")
print("2. Even with same dimensions, vectors are NOT comparable across models")
print("3. Searching with Model B for docs indexed with Model A = POOR RESULTS")
print("4. Similarity scores will be misleading/meaningless")
print("5. ALWAYS use the SAME MODEL for both indexing and searching")
print("=" * 100)
print("\nBest Practice: Store model name with embeddings to ensure consistency")
print("=" * 100)
