import pytest

from runbook_rag.answer import NOT_FOUND, answer_question
from runbook_rag.config import Settings
from runbook_rag.embeddings import HashingEmbeddings
from runbook_rag.evals import run_eval
from runbook_rag.index import FaissIndex
from runbook_rag.ingest import build_index, chunk_documents, load_documents
from runbook_rag.llm import ExtractiveStubModel


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    return Settings(index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture(scope="module")
def index(settings):
    return build_index(settings)


def test_documents_are_loaded_and_chunked(settings):
    documents = load_documents(settings.doc_dir)
    chunks = chunk_documents(documents, size=300, overlap=50)
    assert len(documents) == 6
    assert len(chunks) > len(documents)
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert chunks[0].metadata["chunk_id"].endswith("#1")


def test_overlap_repeats_text_between_neighbouring_chunks(settings):
    document = load_documents(settings.doc_dir)[0]
    chunks = chunk_documents([document], size=200, overlap=60)
    shared = [a.page_content[-20:] in b.page_content for a, b in zip(chunks, chunks[1:])]
    assert any(shared)


def test_embeddings_are_normalised_and_deterministic():
    emb = HashingEmbeddings()
    a, b = emb.embed_query("vpn error 809"), emb.embed_query("vpn error 809")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-5


def test_retrieval_finds_the_right_document(index):
    results = index.search("What does error code 809 mean on the VPN?", k=3)
    assert results[0][0].metadata["source"] == "vpn_setup.md"
    assert results[0][1] >= results[-1][1]  # sorted by similarity


def test_answer_has_citations_and_sources(index):
    answer = answer_question("How many characters must a password have?", index, ExtractiveStubModel())
    assert answer.grounded
    assert "12 characters" in answer.text
    assert "[1]" in answer.text or "[2]" in answer.text
    assert answer.sources[0]["title"] == "Password policy and account lock"


def test_out_of_scope_question_is_refused(index):
    answer = answer_question("What is on the cafeteria menu today?", index, ExtractiveStubModel())
    assert answer.text == NOT_FOUND
    assert not answer.grounded


def test_index_can_be_saved_and_loaded(index, tmp_path):
    index.save(tmp_path)
    loaded = FaissIndex.load(tmp_path, HashingEmbeddings())
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.search("secure print release", k=1)[0][0].metadata["source"] == "printer_setup.md"


def test_eval_reports_metrics(settings, index):
    report = run_eval(settings, verbose=False)
    assert report["answerable_questions"] == 12
    assert 0 <= report["mrr"] <= 1
