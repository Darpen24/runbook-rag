# Runbook RAG

A small retrieval augmented generation (RAG) pipeline over markdown runbooks. It loads
documents, splits them into overlapping chunks, embeds them, stores them in a FAISS index,
retrieves the best chunks for a question and answers with numbered source citations.

I built it as a compact example of the full RAG loop, including a way to measure retrieval.
It is a learning project, not a deployed system.

## How it works

1. **Ingest** (`ingest.py`): read `data/docs/*.md`, split with LangChain's
   `RecursiveCharacterTextSplitter` (500 characters, 80 overlap by default) and give every
   chunk an id such as `vpn_setup.md#3`.
2. **Embed and index** (`embeddings.py`, `index.py`): vectors go into a FAISS
   `IndexFlatIP` on normalised vectors, so the score is cosine similarity. The chunks are
   stored next to the index.
3. **Retrieve** (`index.py`): top-k chunks with their similarity scores.
4. **Answer** (`answer.py`): a LangChain prompt gets the numbered chunks and the question.
   The answer cites sources like `[1]`. If the best similarity is below `MIN_SIMILARITY`,
   the pipeline refuses and does not call the model at all.
5. **Evaluate** (`evals.py`): hit rate at k and mean reciprocal rank (MRR) on labelled
   questions, plus a check that out of scope questions are refused.

## Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows: set PYTHONPATH=src

python -m runbook_rag ingest
python -m runbook_rag ask "What does error code 809 mean on the VPN?"
python -m runbook_rag eval
pytest
```

Defaults need no API key:

- `EMBEDDINGS_PROVIDER=hashing` is an offline stand in that hashes words and word pairs. It
  matches shared words, not meaning, so paraphrased questions will score worse than with a
  real embedding model.
- `LLM_PROVIDER=stub` is **not** a language model. It returns the sentences from the
  retrieved chunks that best match the question.

For real results, copy `.env.example`, set an embedding model and a chat model, and install
the matching package (for example `langchain-openai`). Re-run `ingest` after you change the
embedding model, and re-tune `MIN_SIMILARITY`, because scores are not comparable between
embedding models.

## Results on the demo data

With the default offline settings, on 12 answerable and 2 out of scope questions:
hit rate at 3 is 1.0, MRR is 0.958, and both out of scope questions are refused.

Treat these numbers as a smoke test, not as a quality claim. The question set is tiny,
I wrote it together with the six documents, and the questions share many words with the
text. A real evaluation needs more documents, questions written by other people and
paraphrased questions.

## Limits and next steps

- Retrieval only. There is no reranking, hybrid keyword search or query rewriting.
- Chunks are split by size, not by document structure.
- The out of scope guard is a single similarity threshold.
- Next steps: a real embedding model, a larger labelled set with paraphrases, a reranker
  and checks that the cited chunk really supports the answer.

The documents are made up for the demo.

> These are the original implementation notes. Use the main README for the current installation commands and docs/demo-output.txt for newly verified results.
