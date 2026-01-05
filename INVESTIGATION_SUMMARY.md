# Investigation Summary

## Questions Investigated

### Q1: What happens when database contains 384-dim embeddings but you search with 385-dim embedding?
**Answer:** **PostgreSQL REJECTS the operation immediately**

- Error Code: 22000 (Data Exception)
- Error Message: `expected 1536 dimensions, not 384`
- Happens at: **INSERT time** (cannot even store mismatched dimensions)
- Conclusion: The schema `VECTOR(1536)` enforces STRICT dimension matching

### Q2: What happens when using text-embedding-3-small for indexing but text-embedding-005 for search (both 384 dims)?
**Answer:** **Embeddings are INCOMPATIBLE even with matching dimensions**

- Different models create different vector spaces
- Cosine distances become meaningless
- Search results will be poor/random
- Similarity scores are misleading
- Conclusion: **ALWAYS use the same model for both indexing and search**

---

## Key Findings

### 1. Dimension Enforcement is STRICT
```
PostgreSQL Schema: VECTOR(1536)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
INSERT 384? ❌      INSERT 1536? ✅
REJECTED            ACCEPTED
```

### 2. Model Choice is CRITICAL
```
Index with Model A → [space_A vectors]
Search with Model B → [space_B query]
                    ↓
            POOR RESULTS ❌
            (incompatible spaces)
```

### 3. Current Implementation Issue
- **Schema**: `VECTOR(1536)`
- **Code**: Uses `dimensions=1536`
- **Status**: ✅ Dimensions match correctly
- **Model**: `text-embedding-3-small-1` (produces 1536-dim by default)

---

## Files Created

1. **test_dimension_mismatch.py** - Tests dimension constraints
2. **test_different_models.py** - Tests model compatibility
3. **INVESTIGATION_RESULTS.md** - Detailed findings and recommendations
4. **Updated .github/copilot-instructions.md** - Added critical warnings

---

## Recommendations

### ✅ DO
- Keep dimensions consistent: schema = code = model output
- Use ONE embedding model throughout the entire pipeline
- Document model and dimension choices
- Validate dimensions before DB operations

### ❌ DON'T
- Try to change dimensions without updating schema
- Mix different embedding models
- Assume vectors are compatible because dimensions match
- Reuse embeddings from a different model

---

## Testing Commands

```bash
# Test dimension mismatch scenarios
python test_dimension_mismatch.py

# Test different models scenario
python test_different_models.py

# Verify implementation
python test_implementation.py
```

---

## Critical Rule

**Vector Space Identity = Model + Dimensions**

Change either → All embeddings must be regenerated
