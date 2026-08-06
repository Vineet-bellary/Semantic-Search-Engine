import json
import re
from collections import defaultdict
from pathlib import Path

from semantic_search_engine.config import INGESTED_DATA_DIR, TEST_DIR
from semantic_search_engine.ingestion.representation.embedding import EmbeddingModel
from semantic_search_engine.retrieval.process_query import (
    preprocess_query,
    validate_query,
)
from semantic_search_engine.retrieval.similarity import rank_chunks
from semantic_search_engine.utils.save_load_metadata import load_ingested_data
from semantic_search_engine.process_documents import ingestion


def prepare_query(query: str, embedding_model: EmbeddingModel):
    """Preprocess the query and generate its embedding vector using the embedding model.

    Args:
        query (str): The input query string.
        embedding_model (EmbeddingModel): The embedding model to use.

    Returns:
        list[float]: The embedding vector of the preprocessed query.
    """
    preprocessed_query = preprocess_query(query)
    return embedding_model.embed_query(preprocessed_query)


def load_evaluation_queries(json_path: Path) -> list[dict]:
    """Load evaluation queries from a JSON file.

    Args:
        json_path (Path): The path to the JSON file containing evaluation queries.

    Returns:
        list[dict]: A list of evaluation queries, each represented as a dictionary.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Support both schemas:
    # 1) Legacy: {"evaluation_queries": [{query, expected_document, expected_page_number}, ...]}
    # 2) V3: [{query, expected_document, expected_heading_path}, ...]
    if isinstance(payload, list):
        evaluation_queries = payload
    elif isinstance(payload, dict):
        evaluation_queries = payload.get("evaluation_queries", [])
    else:
        evaluation_queries = []

    if not evaluation_queries:
        raise ValueError(f"No evaluation queries found in {json_path}")

    for i, item in enumerate(evaluation_queries, start=1):
        if "query" not in item or "expected_document" not in item:
            raise ValueError(
                f"Invalid entry at index {i}: each item must contain 'query' and "
                f"'expected_document'."
            )

        has_legacy_target = "expected_page_number" in item
        has_v3_target = "expected_heading_path" in item

        if not has_legacy_target and not has_v3_target:
            raise ValueError(
                f"Invalid entry at index {i}: each item must contain either "
                f"'expected_page_number' (legacy) or 'expected_heading_path' (v3)."
            )

        if has_v3_target:
            heading_path = item["expected_heading_path"]
            if (
                not isinstance(heading_path, list)
                or not heading_path
                or not all(isinstance(h, str) and h.strip() for h in heading_path)
            ):
                raise ValueError(
                    f"Invalid entry at index {i}: 'expected_heading_path' must be a "
                    f"non-empty list of non-empty strings."
                )

    return evaluation_queries


def normalize_document_name(document_name: str) -> str:
    """Normalize document names across legacy PDF names and slug-style identifiers."""
    stem = Path(document_name).stem
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return normalized


def normalize_heading_text(text: str) -> str:
    """Normalize heading text for robust comparisons across formatting variants."""
    return " ".join(text.lower().split())


def heading_path_matches(
    chunk_heading_path: list[str], expected_heading_path: list[str]
) -> bool:
    """Return True if chunk heading path matches expected heading path.

    Matching is robust to representation differences:
    - exact full-path match
    - expected path appearing as a suffix of the chunk path
    - fallback on exact leaf heading match
    """
    if not chunk_heading_path or not expected_heading_path:
        return False

    chunk_norm = [normalize_heading_text(h) for h in chunk_heading_path]
    expected_norm = [normalize_heading_text(h) for h in expected_heading_path]

    if chunk_norm == expected_norm:
        return True

    if (
        len(expected_norm) <= len(chunk_norm)
        and chunk_norm[-len(expected_norm) :] == expected_norm
    ):
        return True

    if expected_norm[-1] == chunk_norm[-1]:
        return True

    return False


def evaluate(evaluation_json_path: Path, k_values: tuple[int, ...] = (1, 3)):
    """
    Evaluate the semantic search engine using a set of evaluation queries.

    Args:
        evaluation_json_path (Path): The path to the JSON file containing evaluation queries.
        k_values (tuple[int, ...], optional): The values of k for Accuracy@k evaluation. Defaults to (1, 3).

    Returns:
        None
    """
    ingestion()

    embedding_model = EmbeddingModel()
    device = embedding_model.device

    chunks, embeddings = load_ingested_data(INGESTED_DATA_DIR, device=device)
    evaluation_queries = load_evaluation_queries(evaluation_json_path)

    total = len(evaluation_queries)
    correct_at_k = {k: 0 for k in k_values}
    per_document_total = defaultdict(int)
    per_document_correct_at_1 = defaultdict(int)
    per_target_total = defaultdict(int)
    per_target_correct_at_1 = defaultdict(int)
    misses = []

    for item in evaluation_queries:
        query = item["query"]
        expected_document = normalize_document_name(item["expected_document"])
        expected_heading_path = item.get("expected_heading_path")
        expected_page = item.get("expected_page_number")
        if expected_page is not None:
            expected_page = int(expected_page)

        validate_query(query)
        query_vector = prepare_query(query, embedding_model)

        max_k = min(max(k_values), len(chunks))
        scores, top_indices = rank_chunks(
            query_vector, embeddings, num_suggestions=max_k
        )

        top_matches = []
        for score, idx in zip(scores, top_indices):
            idx_int = int(idx.item())
            top_matches.append(
                {
                    "document_name": normalize_document_name(
                        chunks[idx_int]["document_name"]
                    ),
                    "page_number": int(chunks[idx_int].get("page_number", -1)),
                    "heading_path": chunks[idx_int].get("heading_path", []),
                    "score": float(score.item()),
                }
            )

        per_document_total[expected_document] += 1

        if expected_heading_path is not None:
            target_key = " > ".join(expected_heading_path)
        else:
            target_key = f"page:{expected_page}"
        per_target_total[target_key] += 1

        def is_correct_match(match: dict) -> bool:
            if match["document_name"] != expected_document:
                return False

            if expected_heading_path is not None:
                return heading_path_matches(
                    match.get("heading_path", []), expected_heading_path
                )

            return match.get("page_number") == expected_page

        for k in k_values:
            k_eff = min(k, len(top_matches))
            if any(is_correct_match(match) for match in top_matches[:k_eff]):
                correct_at_k[k] += 1

        if top_matches and is_correct_match(top_matches[0]):
            per_target_correct_at_1[target_key] += 1
            per_document_correct_at_1[expected_document] += 1
        else:
            predicted_document = (
                top_matches[0]["document_name"] if top_matches else None
            )
            predicted_page = top_matches[0]["page_number"] if top_matches else None
            predicted_heading_path = (
                top_matches[0]["heading_path"] if top_matches else None
            )
            misses.append(
                {
                    "query": query,
                    "expected_document": expected_document,
                    "expected_heading_path": expected_heading_path,
                    "expected_page_number": expected_page,
                    "predicted_document_top1": predicted_document,
                    "predicted_heading_path_top1": predicted_heading_path,
                    "predicted_page_number_top1": predicted_page,
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

    print("\nPer-target Accuracy@1")
    print("-" * 100)
    for target in sorted(per_target_total):
        target_total = per_target_total[target]
        target_correct = per_target_correct_at_1[target]
        print(
            f"{target}: {target_correct / target_total:.2%} ({target_correct}/{target_total})"
        )

    print("\nPer-document Accuracy@1")
    print("-" * 100)
    for document_name in sorted(per_document_total):
        document_total = per_document_total[document_name]
        document_correct = per_document_correct_at_1[document_name]
        print(
            f"{document_name}: {document_correct / document_total:.2%} "
            f"({document_correct}/{document_total})"
        )

    print("\nTop-1 Misses")
    print("-" * 100)
    if not misses:
        print("No misses. Perfect top-1 accuracy.")
    else:
        for miss in misses:
            expected_target = (
                f"heading {' > '.join(miss['expected_heading_path'])}"
                if miss["expected_heading_path"]
                else f"page {miss['expected_page_number']}"
            )
            predicted_target = (
                f"heading {' > '.join(miss['predicted_heading_path_top1'])}"
                if miss["predicted_heading_path_top1"]
                else f"page {miss['predicted_page_number_top1']}"
            )
            print(
                f"Query: {miss['query']}\n"
                f"Expected: {miss['expected_document']} {expected_target}\n"
                f"Predicted Top-1: {miss['predicted_document_top1']} {predicted_target}\n"
                f"Top-k matches: {miss['top_k_matches']}\n" + "-" * 100
            )


def main():
    evaluation_json_path = TEST_DIR / "evaluation_queries.json"
    evaluate(evaluation_json_path, k_values=(1, 3))


if __name__ == "__main__":
    main()
