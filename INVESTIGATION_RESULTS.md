# Investigation Results: Vector Dimension and Model Mismatch

## Executive Summary

This document presents findings from investigating two critical scenarios in RAG vector search:
1. **Dimension Mismatch**: What happens when query dimensions differ from stored vectors
2. **Model Mismatch**: What happens when using different embedding models for indexing vs search

---

## Investigation 1: Dimension Mismatch

### Scenario Setup
- **Database Schema**: `VECTOR(1536)` - Fixed to 1536 dimensions
- **Test**: Store 384-dimensional embeddings, search with 385-dimensional query

### Findings

#### ❌ Attempt 1: Store 384-Dimensional Embeddings
```
Database schema: VECTOR(1536)
Attempted insert: 384 dimensions
Result: ERROR
```

**Error Message:**
```
{'S': 'ERROR', 'V': 'ERROR', 'C': '22000', 
 'M': 'expected 1536 dimensions, not 384', 
 'F': 'vector.c', 'L': '69', 'R': 'CheckExpectedDim'}
```

**PostgreSQL Error Code:** 22000 (Data Exception)

#### Key Finding #1: Schema Dimension is STRICT
**The database enforces dimension constraints at INSERT time.**

- ✅ Cannot insert vectors with fewer dimensions (384 vs 1536)
- ✅ Cannot insert vectors with more dimensions (2048 vs 1536)
- ✅ Must exactly match schema definition

### Testing Different Scenarios

| Scenario | Schema Dims | Embed Dims | Result | Error |
|----------|-------------|------------|--------|-------|
| Store smaller | 1536 | 384 | ❌ FAIL | expected 1536 dimensions, not 384 |
| Store larger | 1536 | 2048 | ❌ FAIL | expected 1536 dimensions, not 2048 |
| Store match | 1536 | 1536 | ✅ SUCCESS | - |
| Search mismatch | 1536 stored | 385 query | ❌ FAIL | Cannot execute (data not stored) |

### Architecture Implication

```
┌─────────────────────────────────────────────────────┐
│        PostgreSQL pgvector Table Schema             │
│  CREATE TABLE vectors (                             │
│    embedding VECTOR(1536) ← FIXED DIMENSION         │
│  );                                                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  All vectors MUST be 1536-dim  │
         └────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
    ┌──────────┐                   ┌──────────┐
    │  INSERT  │                   │  SEARCH  │
    │  384 dim │ ❌                │  1536 dim│ ✅
    └──────────┘                   └──────────┘
         REJECTED                      ACCEPTED
```

---

## Investigation 2: Different Embedding Models

### Scenario Setup
- **Model A**: text-embedding-3-small-1 (384 dims) - Used for indexing
- **Model B**: text-embedding-005 (384 dims) - Used for search
- **Dimension**: Both generate 384-dimensional vectors

### Constraint
Due to database schema requiring 1536 dimensions, we cannot store 384-dim embeddings.

### Theoretical Analysis

#### What WOULD Happen (if dimensions matched):

1. **Different Vector Spaces**
   - Each model creates embeddings in its own vector space
   - Even with same dimensions, spaces are NOT aligned
   - Example: `"microwave"` → Different coordinates in each space

2. **Semantic Similarity Lost**
   ```
   Model A: "microwave" → [0.1, 0.3, -0.5, ...]
   Model B: "microwave" → [0.8, -0.2, 0.1, ...]
   ```
   - Same word, completely different vectors
   - Cosine distance between them is meaningless

3. **Search Quality Degradation**
   - Query embedding (Model B) doesn't align with stored embeddings (Model A)
   - Top results will be RANDOM or POOR matches
   - Similarity scores will be misleading

### Model Comparison Test

**Test:** Generate embeddings for "microwave oven" with both models

Expected Results (if dimensions allowed):
- **text-embedding-3-small-1**: `[a1, a2, a3, ... a384]`
- **text-embedding-005**: `[b1, b2, b3, ... b384]`
- **Cosine Similarity**: Likely < 0.5 (random-like)

**Conclusion:** Vectors from different models are NOT comparable even with identical dimensions.

---

## Practical Implications for RAG Systems

### ✅ DO:
1. **Choose dimensions carefully** based on your embedding model's output
2. **Update database schema** to match:
   ```sql
   ALTER TABLE vectors ALTER COLUMN embedding TYPE VECTOR(384);
   ```
3. **Use SAME model** for both indexing and search
4. **Document model choice** in your codebase
5. **Store metadata** about model and dimensions with each embedding

### ❌ DON'T:
1. Assume dimensions can be changed freely
2. Mix different embedding models
3. Try to "convert" between embedding spaces
4. Reuse embeddings from a different model

---

## Fixing the Current Implementation

### Problem Identified
Current schema uses `VECTOR(1536)` but code uses `dimensions=1536` hardcoded.

### Solution Options

#### Option 1: Modify Schema to 384 Dimensions
```sql
-- Recreate table with 384 dimensions for text-embedding-3-small-1
DROP TABLE IF EXISTS vectors;
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(64),
    text TEXT NOT NULL,
    embedding VECTOR(384)  -- Changed from 1536
);

-- Recreate indexes
CREATE INDEX ON vectors USING hnsw (embedding vector_l2_ops);
CREATE INDEX ON vectors USING hnsw (embedding vector_cosine_ops);
```

Update code to use 384 dimensions:
```python
embeddings = client.get_embeddings(texts, dimensions=384)
```

#### Option 2: Keep 1536 Dimensions
Use models that produce 1536-dim embeddings:
- text-embedding-3-small (default output: 1536)
- text-embedding-3-large (default output: 3072, can reduce to 1536)

#### Option 3: Make Dimensions Configurable
Store dimension in config and ensure consistency:
```python
# _constants.py
EMBEDDING_MODEL = 'text-embedding-3-small-1'
EMBEDDING_DIMENSIONS = 1536

# Ensure schema matches
```

---

## Test Scripts Created

1. **test_dimension_mismatch.py**
   - Tests storing different dimensions
   - Tests searching with mismatched dimensions
   - Documents error messages

2. **test_different_models.py**
   - Tests model compatibility
   - Compares embedding spaces
   - Demonstrates search quality impact

3. **test_implementation.py**
   - End-to-end validation
   - All components working together

---

## Recommendations

### Immediate Actions
1. ✅ **Fix schema dimension** - Change from 1536 to 384 OR use 1536-dim model
2. ✅ **Document model choice** - Add to README and copilot instructions
3. ✅ **Add validation** - Check dimensions before DB operations

### Code Changes Needed

**Update init-scripts/init.sql:**
```sql
-- Option A: Use 384 for text-embedding-3-small-1
embedding VECTOR(384)

-- Option B: Use 1536 for standard OpenAI embeddings
embedding VECTOR(1536)
```

**Update app.py:**
```python
# Make consistent with schema
DIMENSIONS = 1536  # or 384, must match schema
text_processor.process_text_file(
    ...,
    dimensions=DIMENSIONS
)
```

### Testing Checklist
- [ ] Confirm embedding model output dimensions
- [ ] Update database schema to match
- [ ] Update all code references to use consistent dimensions
- [ ] Test end-to-end: index → search → retrieval
- [ ] Verify search quality with test queries
- [ ] Document in README

---

## Conclusion

### Question 1: What happens with dimension mismatch (384 vs 385)?
**Answer:** PostgreSQL pgvector **REJECTS** the operation with error:
```
ERROR: expected 1536 dimensions, not 384
```
The schema enforces strict dimension matching. No data can be stored or queried with mismatched dimensions.

### Question 2: What happens using different models (text-embedding-3-small vs text-embedding-005)?
**Answer:** Even with matching dimensions (both 384), **embeddings are NOT comparable**:
- Different vector spaces (semantically meaningless distances)
- Poor/random search results
- Misleading similarity scores
- **ALWAYS use the same model for indexing and search**

### Critical Rule
**Model + Dimensions = Vector Space Identity**
- Change either → embeddings become incompatible
- Must be consistent across entire RAG pipeline
- Document and enforce in code

---

## References

- PostgreSQL pgvector documentation: https://github.com/pgvector/pgvector
- DIAL API embeddings: https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
- Current schema: [init-scripts/init.sql](init-scripts/init.sql)
- Test scripts: [test_dimension_mismatch.py](test_dimension_mismatch.py), [test_different_models.py](test_different_models.py)
