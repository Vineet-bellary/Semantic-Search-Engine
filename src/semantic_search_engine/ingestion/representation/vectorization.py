from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_chunks(chunks: list[dict]) -> tuple:
    text_chunks_corpus = [chunk.get("text_chunk", "") for chunk in chunks]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_chunks_corpus)

    return vectors, vectorizer
