import re

def preprocess_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query

def validate_query(query: str) -> bool:
    if not query:
        raise ValueError("Query cannot be empty.")
    if len(query) > 200:
        raise ValueError("Query is too long. Please limit it to 200 characters.")
    return True

def vectorize_query(query: str, vectorizer) -> list[float]:
    query_vector = vectorizer.transform([query])
    return query_vector