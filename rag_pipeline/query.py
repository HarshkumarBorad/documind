"""CLI for the DocuMind RAG pipeline.

Single-domain:
    python -m rag_pipeline.query --domain hr --question "What is the leave policy?"

Federated (searches all four namespaces, re-ranks merged results):
    python -m rag_pipeline.query --federated --question "What does the company say about GDPR?"

Useful flags:
    --model <hf_model>        Pick from `--list-models`
    --provider <hf_provider>  Pin a specific HF Inference Provider
    --top-k N                 Single-mode: total results. Federated: per-domain results.
"""
from __future__ import annotations

import argparse
import sys

from rag_pipeline.graph import get_graph
from rag_pipeline.llm import DEFAULT_LLM, SUPPORTED_LLMS
from vectorstore import Domain


def main() -> int:
    parser = argparse.ArgumentParser(description="Query a DocuMind knowledge namespace.")
    parser.add_argument("--domain", choices=[d.value for d in Domain])
    parser.add_argument(
        "--federated",
        action="store_true",
        help="Search all four namespaces in parallel and re-rank the merged results.",
    )
    parser.add_argument("--question")
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM,
        help=f"HF chat model. Default: {DEFAULT_LLM}",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--provider",
        default=None,
        help="HF Inference Provider override (auto / together / fireworks-ai / novita / ...). "
        "Default: HF_LLM_PROVIDER env var, otherwise 'auto'.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the supported model list and exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Supported LLMs (set as --model):")
        for m in SUPPORTED_LLMS:
            marker = "  (default)" if m == DEFAULT_LLM else ""
            print(f"  - {m}{marker}")
        return 0

    if not args.question:
        parser.error("--question is required (unless --list-models)")
    if not args.federated and not args.domain:
        parser.error("Either --domain (single-mode) or --federated must be specified.")
    if args.federated and args.domain:
        parser.error("--domain and --federated are mutually exclusive.")

    if args.federated:
        print("Mode    : federated (all 4 namespaces)")
        print(f"Model   : {args.model}")
        print(f"Top-K   : {args.top_k} per domain")
    else:
        print(f"Mode    : single ({args.domain})")
        print(f"Model   : {args.model}")
        print(f"Top-K   : {args.top_k}")
    print(f"Question: {args.question}\n")

    graph_input = {
        "question": args.question,
        "llm_name": args.model,
    }
    if args.federated:
        graph_input["query_mode"] = "federated"
        graph_input["top_k_per_domain"] = args.top_k
    else:
        graph_input["query_mode"] = "single"
        graph_input["domain"] = args.domain
        graph_input["top_k"] = args.top_k
    if args.provider:
        graph_input["llm_provider"] = args.provider

    try:
        result = get_graph().invoke(graph_input)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(result.get("answer", "(no answer)"))
    print("=" * 60)

    retrieved = result.get("retrieved", [])
    sources = result.get("sources", [])
    print(
        f"\nRetrieved {len(retrieved)} chunk(s) post-rerank; LLM cited {len(sources)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
