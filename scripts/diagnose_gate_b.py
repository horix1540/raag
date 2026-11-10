"""Diagnose Gate B failures: cases where answer is correct but NLI flags contradiction.

Rebuilds the exact retrieval+generation flow for specific items to show what text
DeBERTa evaluated and why it flagged a contradiction.
"""

import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import index_texts
from src.pipeline import answer_question
from src.squad_metrics import exact_match_score, f1_score


def diagnose_item(item_id: str, data_path: str = "data/benchmarks/hotpot_dev_150.jsonl"):
    """Reconstruct the pipeline for one item and print Gate B's reasoning."""
    # Load the item
    with open(data_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    item = next((x for x in items if x["item_id"] == item_id), None)
    if not item:
        print(f"Item {item_id} not found")
        return

    print(f"\n{'='*80}")
    print(f"Item: {item_id}")
    print(f"Question: {item['question']}")
    print(f"Gold answers: {item['answers']}")
    print(f"{'='*80}\n")

    # Build chunk-level index
    view = item["chunk_view"]
    units = view["units"]
    gold_indices = set(view["gold_units"])

    collection_name = f"diagnose_{item_id}_chunk"
    store = index_texts(units, collection=collection_name, source="hotpot:chunk")

    # Run baseline pipeline
    result = answer_question(
        item["question"],
        store=store,
        collection=collection_name,
        baseline=True,  # Gates measured but not enforced
    )

    answer = result.answer or ""
    em = exact_match_score(answer, item["answers"])
    f1 = f1_score(answer, item["answers"])

    print(f"Generated answer: {answer!r}")
    print(f"EM: {em:.3f}  F1: {f1:.3f}")
    print()

    # Show Gate B evaluation
    if result.confidence.gate_b:
        gate_b = result.confidence.gate_b
        print(f"Gate B (NLI faithfulness): score={gate_b.score:.6f}  passed={gate_b.passed}")
        print()

        # The NLI check in gate_b evaluates: answer (hypothesis) vs retrieved chunks (premise)
        # Let's show what premises were evaluated
        print("Retrieved chunks evaluated by Gate B:")
        print("-" * 80)

        for i, chunk in enumerate(result.retrieval.chunks, 1):
            chunk_idx = chunk.metadata.get("chunk_index", -1)
            is_gold = chunk_idx in gold_indices
            print(f"\n[Chunk {i}/5] (index={chunk_idx}, gold={'YES' if is_gold else 'NO'})")
            text = chunk.text.encode('ascii', errors='replace').decode('ascii')
            print(f"{text[:500]}..." if len(text) > 500 else text)
    else:
        print("Gate B was not evaluated (retrieval likely failed Gate A)")

    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("Does the answer appear in the retrieved text? Check the chunks above.")
    print("Is the contradiction legitimate, or is DeBERTa confused by:")
    print("  1. Multi-hop: Answer requires connecting facts across different chunks?")
    print("  2. Padding: Chunk contains irrelevant sentences that contradict the answer?")
    print("  3. Implicit reasoning: Answer requires inference not explicitly stated?")
    print("="*80 + "\n")

    # Cleanup
    with contextlib.suppress(Exception):
        store._client.delete_collection(collection_name)


if __name__ == "__main__":
    # The 5 cases with high F1 but hard contradiction from the query above
    problem_items = [
        "5aba9e2455429939ce03dd3a",  # Rejuvelac/1909: F1=1.0, NLI=0.0008
        "5abffc41554299012d1db551",  # Marlene/1901: F1=1.0, NLI=0.0005
        "5a7363ec5542991f29ee2dd7",  # USS Tortuga/Key West: F1=0.67, NLI=0.0022
        "5ab3b2c4554299233954ff8a",  # Chapter 27/Catcher: F1=0.6, NLI=0.004
        "5a89e0ec5542992e4fca8427",  # Howard Roark/Objectivism: F1=1.0, NLI=0.15
    ]

    for item_id in problem_items:
        diagnose_item(item_id)
        input("\nPress Enter to continue to next case...\n")
