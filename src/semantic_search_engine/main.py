from semantic_search_engine.config import INGESTED_DATA_DIR

from semantic_search_engine import process_documents, search


def main():
    # if not (INGESTED_DATA_DIR.exists() and any(INGESTED_DATA_DIR.iterdir())):
    #     process_documents.ingestion()
    process_documents.ingestion()
    search.search()


if __name__ == "__main__":
    main()
