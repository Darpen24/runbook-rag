from __future__ import annotations

import argparse

from .answer import answer_question
from .config import Settings
from .evals import run_eval
from .ingest import build_index, load_index
from .llm import build_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Small RAG pipeline over markdown runbooks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Chunk, embed and index the documents")
    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    sub.add_parser("eval", help="Measure retrieval quality")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.command == "ingest":
        index = build_index(settings)
        print(f"Indexed {len(index.chunks)} chunks into {settings.index_dir}")
    elif args.command == "ask":
        index = load_index(settings)
        llm = build_llm(settings.llm_provider, settings.llm_model)
        result = answer_question(args.question, index, llm, settings.top_k, settings.min_similarity)
        print(result.text)
        for source in result.sources:
            print(f"  [{source['n']}] {source['chunk_id']} ({source['title']}) similarity {source['similarity']}")
    else:
        run_eval(settings)


if __name__ == "__main__":
    main()
