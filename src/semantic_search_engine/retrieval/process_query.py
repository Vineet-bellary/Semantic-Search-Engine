import re


def preprocess_query(query: str) -> str:
    """Preprocess the query by converting it to lowercase, removing special characters, and normalizing whitespace."""
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def validate_query(query: str) -> bool:
    """Validate the query to ensure it is not empty and does not exceed a certain length."""
    if not query:
        raise ValueError("Query cannot be empty.")
    if len(query) > 200:
        raise ValueError("Query is too long. Please limit it to 200 characters.")
    return True


def vectorize_query(query: str, vectorizer) -> list[float]:
    """Vectorize the query using the provided vectorizer.

    Args:
        query (str): The input query string.
        vectorizer: The vectorizer to use for transforming the query.

    Returns:
        list[float]: The vectorized representation of the query.
    """
    query_vector = vectorizer.transform([query])
    return query_vector
