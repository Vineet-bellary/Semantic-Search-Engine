import json
from collections import defaultdict
from pathlib import Path

from config import DATA_DIR, INGESTED_DATA_DIR
from retrieval.process_query import preprocess_query, validate_query, vectorize_query
from retrieval.similarity import rank_chunks
from utils.save_load_metadata import load_ingested_data


def prepare_query(query: str, vectorizer):
    preprocessed_query = preprocess_query(query)
    return vectorize_query(preprocessed_query, vectorizer)


def load_evaluation_queries(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    evaluation_queries = payload.get("evaluation_queries", [])
    if not evaluation_queries:
        raise ValueError(f"No evaluation queries found in {json_path}")

    for i, item in enumerate(evaluation_queries, start=1):
        if (
            "query" not in item
            or "expected_document" not in item
            or "expected_page_number" not in item
        ):
            raise ValueError(
                f"Invalid entry at index {i}: each item must contain 'query', 'expected_document', and 'expected_page_number'."
            )

    return evaluation_queries


def normalize_document_name(document_name: str) -> str:

    return Path(document_name).name


def evaluate(evaluation_json_path: Path, k_values: tuple[int, ...] = (1, 3)):
    if not (INGESTED_DATA_DIR.exists() and any(INGESTED_DATA_DIR.iterdir())):
        raise FileNotFoundError(
            f"Ingested data not found in {INGESTED_DATA_DIR}. Run ingestion first."
        )

    chunks, chunks_vectors, vectorizer = load_ingested_data(INGESTED_DATA_DIR)
    evaluation_queries = load_evaluation_queries(evaluation_json_path)

    total = len(evaluation_queries)
    correct_at_k = {k: 0 for k in k_values}
    per_page_total = defaultdict(int)
    per_page_correct_at_1 = defaultdict(int)
    per_document_total = defaultdict(int)
    per_document_correct_at_1 = defaultdict(int)
    misses = []

    for item in evaluation_queries:
        query = item["query"]
        expected_document = normalize_document_name(item["expected_document"])
        expected_page = int(item["expected_page_number"])

        validate_query(query)
        query_vector = prepare_query(query, vectorizer)

        max_k = max(k_values)
        scores, top_indices = rank_chunks(
            query_vector, chunks_vectors, num_suggestions=max_k
        )
        top_matches = [
            {
                "document_name": normalize_document_name(chunks[idx]["document_name"]),
                "page_number": int(chunks[idx]["page_number"]),
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]

        per_page_total[expected_page] += 1
        per_document_total[expected_document] += 1

        for k in k_values:
            if any(
                match["document_name"] == expected_document
                and match["page_number"] == expected_page
                for match in top_matches[:k]
            ):
                correct_at_k[k] += 1

        if (
            top_matches[0]["document_name"] == expected_document
            and top_matches[0]["page_number"] == expected_page
        ):
            per_page_correct_at_1[expected_page] += 1
            per_document_correct_at_1[expected_document] += 1
        else:
            misses.append(
                {
                    "query": query,
                    "expected_document": expected_document,
                    "expected_page_number": expected_page,
                    "predicted_document_top1": top_matches[0]["document_name"],
                    "predicted_page_number_top1": top_matches[0]["page_number"],
                    "top_k_matches": top_matches,
                }
            )

    print("\n" + "=" * 100)
    print("Evaluation Results")
    print("=" * 100)
    print(f"Total queries: {total}")
    for k in k_values:
        print(
            f"Accuracy@{k}: {correct_at_k[k] / total:.2%} ({correct_at_k[k]}/{total})"
        )

    print("\nPer-page Accuracy@1")
    print("-" * 100)
    for page in sorted(per_page_total):
        page_total = per_page_total[page]
        page_correct = per_page_correct_at_1[page]
        print(
            f"Page {page}: {page_correct / page_total:.2%} ({page_correct}/{page_total})"
        )

    print("\nPer-document Accuracy@1")
    print("-" * 100)
    for document_name in sorted(per_document_total):
        document_total = per_document_total[document_name]
        document_correct = per_document_correct_at_1[document_name]
        print(
            f"{document_name}: {document_correct / document_total:.2%} ({document_correct}/{document_total})"
        )

    print("\nTop-1 Misses")
    print("-" * 100)
    if not misses:
        print("No misses. Perfect top-1 accuracy.")
    else:
        for miss in misses:
            print(
                f"Query: {miss['query']}\n"
                f"Expected: {miss['expected_document']} page {miss['expected_page_number']}\n"
                f"Predicted Top-1: {miss['predicted_document_top1']} page {miss['predicted_page_number_top1']}\n"
                f"Top-k matches: {miss['top_k_matches']}\n" + "-" * 100
            )


def main():
    evaluation_json_path = DATA_DIR / "evaluation_queries.json"
    evaluate(evaluation_json_path, k_values=(1, 3))


if __name__ == "__main__":
    main()
