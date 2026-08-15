from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static
from textual.worker import Worker, WorkerState

from semantic_search_engine.retrieval.service import SearchResult, SearchService


class SearchApp(App[None]):
    """Keyboard-first Textual interface for semantic document search."""

    TITLE = "Semantic Search Engine"
    SUB_TITLE = "Local document retrieval"

    CSS = """
    Screen {
        layout: vertical;
    }

    #query {
        dock: top;
        margin: 1 2;
    }

    #status {
        height: 1;
        margin: 0 2;
        color: $text-muted;
    }

    #workspace {
        height: 1fr;
        margin: 0 2 1 2;
    }

    #results-pane {
        width: 38%;
        border: round $primary;
        padding: 1;
    }

    #detail-pane {
        width: 62%;
        border: round $secondary;
        padding: 1 2;
        overflow-y: auto;
    }

    ListView {
        height: 1fr;
    }

    ListItem {
        padding: 1;
        height: auto;
    }

    .result-title {
        text-style: bold;
    }

    .result-meta {
        color: $text-muted;
    }

    #detail {
        height: auto;
    }

    """

    BINDINGS = [
        ("ctrl+l", "focus_query", "Focus query"),
        ("ctrl+r", "repeat_search", "Search again"),
        ("escape", "clear_selection", "Clear selection"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.service: SearchService | None = None
        self.results: list[SearchResult] = []
        self.last_query = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Ask a question about your documents...", id="query")
        yield Static("Loading models and search artifacts...", id="status")
        with Horizontal(id="workspace"):
            with Vertical(id="results-pane"):
                yield Label("Results", classes="result-title")
                yield ListView(id="results")
            with Vertical(id="detail-pane"):
                yield Static("Select a result to inspect its content.", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#query", Input).focus()
        self.load_service()

    def on_resize(self, event) -> None:
        workspace = self.query_one("#workspace")
        results_pane = self.query_one("#results-pane")
        detail_pane = self.query_one("#detail-pane")
        if event.size.width < 90:
            workspace.styles.layout = "vertical"
            results_pane.styles.width = "100%"
            detail_pane.styles.width = "100%"
        else:
            workspace.styles.layout = "horizontal"
            results_pane.styles.width = "38%"
            detail_pane.styles.width = "62%"

    def load_service(self) -> None:
        self.query_one("#status", Static).update(
            "Loading models and search artifacts..."
        )
        self._load_service()

    @staticmethod
    def _build_service() -> SearchService:
        return SearchService()

    @work(thread=True, exclusive=True)
    def _load_service(self) -> None:
        service = self._build_service()
        self.call_from_thread(self._service_loaded, service)

    def _service_loaded(self, service: SearchService) -> None:
        self.service = service
        self.query_one("#status", Static).update(
            "Ready. Enter a query and press Enter."
        )

    def _service_failed(self, worker: Worker[SearchService]) -> None:
        self.query_one("#status", Static).update(
            f"Unable to load search: {worker.error}"
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "_load_service" and event.state == WorkerState.ERROR:
            self._service_failed(event.worker)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            self.query_one("#status", Static).update("Enter a query first.")
            return
        self.last_query = query
        self.run_search(query)

    def run_search(self, query: str) -> None:
        if self.service is None:
            self.query_one("#status", Static).update("Search is still loading.")
            return
        self.query_one("#status", Static).update(
            "Searching 20 candidates and reranking..."
        )
        self._search(query)

    @work(thread=True, exclusive=True)
    def _search(self, query: str) -> None:
        results = self.service.search(query) if self.service else []
        self.call_from_thread(self._show_results, results)

    def _show_results(self, results: list[SearchResult]) -> None:
        self.results = results
        result_list = self.query_one("#results", ListView)
        result_list.clear()

        if not results:
            self.query_one("#status", Static).update("No results found.")
            self.query_one("#detail", Static).update("No matching chunks were found.")
            return

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            heading = " > ".join(chunk.headings) if chunk.headings else "No heading"
            pages = ", ".join(str(page) for page in chunk.pages) or "Unknown page"
            result_list.append(
                ListItem(
                    Label(f"{rank}. {chunk.document_name}", classes="result-title"),
                    Label(heading, classes="result-meta"),
                    Label(
                        f"Pages {pages} | Reranker {result.rerank_score:.2f}",
                        classes="result-meta",
                    ),
                )
            )

        self.query_one("#status", Static).update(
            f"{len(results)} results found. Use Up/Down to inspect them."
        )
        result_list.index = 0
        self._show_detail(0)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._show_detail(event.list_view.index)

    def _show_detail(self, index: int | None) -> None:
        if index is None or not 0 <= index < len(self.results):
            return
        result = self.results[index]
        chunk = result.chunk
        heading = " > ".join(chunk.headings) if chunk.headings else "N/A"
        pages = ", ".join(str(page) for page in chunk.pages) or "N/A"
        detail = (
            f"Document: {chunk.document_name}\n"
            f"Heading: {heading}\n"
            f"Pages: {pages}\n"
            f"Retrieval score: {result.retrieval_score:.3f}\n"
            f"Reranker score: {result.rerank_score:.3f}\n\n"
            f"{chunk.content}"
        )
        self.query_one("#detail", Static).update(detail)

    def action_focus_query(self) -> None:
        self.query_one("#query", Input).focus()

    def action_repeat_search(self) -> None:
        if self.last_query:
            self.run_search(self.last_query)

    def action_clear_selection(self) -> None:
        self.query_one("#detail", Static).update(
            "Select a result to inspect its content."
        )


def run_app() -> None:
    SearchApp().run()


if __name__ == "__main__":
    run_app()
