"""
Test script for RAG implementation
Run this to verify all components work correctly
"""
import os
import sys

print("=" * 100)
print("RAG IMPLEMENTATION TEST")
print("=" * 100)

# Test 1: Check imports
print("\n[TEST 1] Checking imports...")
try:
    import pg8000
    from task._constants import API_KEY
    from task.embeddings.embeddings_client import DialEmbeddingsClient
    from task.embeddings.text_processor import TextProcessor, SearchMode
    from task.chat.chat_completion_client import DialChatCompletionClient
    from task.models.message import Message
    from task.models.role import Role
    print("[PASS] All imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Check database connection
print("\n[TEST 2] Checking database connection...")
try:
    conn = pg8000.connect(
        host='localhost',
        port=5433,
        database='vectordb',
        user='postgres',
        password='postgres'
    )
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM vectors")
        count = cur.fetchone()[0]
        print(f"[PASS] Database connected. Vectors table has {count} records")
    conn.close()
except Exception as e:
    print(f"[FAIL] Database connection failed: {e}")
    print("Make sure PostgreSQL container is running: wsl docker ps")
    sys.exit(1)

# Test 3: Check API key
print("\n[TEST 3] Checking DIAL API key...")
if not API_KEY or not API_KEY.strip():
    print("[FAIL] API key is not set!")
    print("Please set DIAL_API_KEY:")
    print("  Option 1: Create .env file with: DIAL_API_KEY=your_key_here")
    print("  Option 2: Set environment variable: $env:DIAL_API_KEY='your_key_here'")
    print("  Option 3: Update task/_constants.py directly")
    print("\nGet your API key from: https://support.epam.com/ess?id=sc_cat_item&table=sc_cat_item&sys_id=910603f1c3789e907509583bb001310c")
    sys.exit(1)
else:
    print(f"[PASS] API key is set (length: {len(API_KEY)} chars)")

# Test 4: Test embeddings client initialization
print("\n[TEST 4] Testing embeddings client...")
try:
    embeddings_client = DialEmbeddingsClient(
        deployment_name='text-embedding-3-small-1',
        api_key=API_KEY
    )
    print("[PASS] Embeddings client initialized")
except Exception as e:
    print(f"[FAIL] Embeddings client error: {e}")
    sys.exit(1)

# Test 5: Test embeddings API call (small test)
print("\n[TEST 5] Testing embeddings API call...")
print("(This will make a real API call - make sure EPAM VPN is connected)")
try:
    test_embeddings = embeddings_client.get_embeddings(
        inputs=["test"],
        dimensions=1536
    )
    if 0 in test_embeddings and len(test_embeddings[0]) == 1536:
        print(f"[PASS] Embeddings API working. Generated {len(test_embeddings[0])}-dimensional vector")
    else:
        print("[FAIL] Unexpected embeddings format")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Embeddings API error: {e}")
    print("Make sure:")
    print("  1. EPAM VPN is connected")
    print("  2. API key is valid")
    sys.exit(1)

# Test 6: Test text processor
print("\n[TEST 6] Testing text processor...")
try:
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    text_processor = TextProcessor(
        embeddings_client=embeddings_client,
        db_config=db_config
    )
    print("[PASS] Text processor initialized")
except Exception as e:
    print(f"[FAIL] Text processor error: {e}")
    sys.exit(1)

# Test 7: Test chat completion client
print("\n[TEST 7] Testing chat completion client...")
try:
    chat_client = DialChatCompletionClient(
        deployment_name='gpt-4o',
        api_key=API_KEY
    )
    print("[PASS] Chat completion client initialized")
except Exception as e:
    print(f"[FAIL] Chat completion client error: {e}")
    sys.exit(1)

print("\n" + "=" * 100)
print("ALL TESTS PASSED!")
print("=" * 100)
print("\nYour RAG implementation is ready to use!")
print("\nNext steps:")
print("  1. Run the application: python -m task.app")
print("  2. Choose 'y' when asked to load context")
print("  3. Ask questions about microwave usage")
print("\nValid test questions:")
print('  - "What safety precautions should be taken?"')
print('  - "How should you clean the glass tray?"')
print('  - "What is the maximum cooking time?"')
print("\nInvalid test questions (should be rejected):")
print('  - "What do you know about DIALX community?"')
print('  - "Tell me about dinosaurs"')
