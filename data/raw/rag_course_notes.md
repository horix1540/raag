# Retrieval-Augmented Generation: Course Notes

## Unit 1 — What RAG Is

Retrieval-Augmented Generation (RAG) combines a parametric language model with a
non-parametric external memory. Instead of relying only on weights learned during
pre-training, the system retrieves relevant passages from a document collection at
query time and conditions generation on those passages. The approach was introduced
by Lewis et al. in 2020.

The immediate benefit is that the knowledge base can be updated without retraining
the model. Adding a new document to the index makes that information available on the
very next query.

## Unit 2 — The Standard Pipeline

A conventional RAG pipeline has five stages:

1. **Ingestion** — documents are loaded and normalised into plain text.
2. **Chunking** — long documents are split into passages, typically 200 to 1000
   characters, usually with some overlap so a sentence spanning a boundary is not
   lost.
3. **Embedding** — each chunk is mapped to a dense vector by an encoder such as
   all-MiniLM-L6-v2, which produces 384-dimensional embeddings.
4. **Retrieval** — the query is embedded with the same encoder and the nearest
   chunks are found by cosine similarity, usually via an approximate index such as
   HNSW.
5. **Generation** — the retrieved chunks are placed in the prompt and a language
   model produces the answer.

## Unit 3 — Why RAG Still Hallucinates

Retrieval reduces hallucination but does not eliminate it. Three distinct failure
modes are worth separating.

First, **retrieval failure**: the corpus simply does not contain the answer. A
similarity search always returns its top-k nearest neighbours, so the generator
receives passages regardless of whether any of them are relevant. Nearest does not
mean sufficient.

Second, **grounding failure**: the correct passage is retrieved but the model still
produces an unsupported statement, filling a gap from parametric memory rather than
admitting the gap exists.

Third, **conflict**: the model's output directly contradicts the retrieved passage.

The RAGTruth dataset distinguishes the last two cases as *Baseless Information*,
where a claim is simply absent from the source, and *Evident Conflict*, where a claim
disagrees with the source. Evident Conflict is considered the more severe failure.

## Unit 4 — Evaluation Metrics

Faithfulness measures whether every claim in the generated answer is supported by the
retrieved context. Answer relevancy measures whether the answer addresses the
question. Contextual relevancy measures whether the retrieved passages are pertinent
to the query, and is a property of the retriever rather than the generator.

Because faithfulness requires an answer to already exist, it can only be computed
after generation. Contextual relevancy needs only the query and the passages, so it
can be computed before generation.

## Unit 5 — Selective Prediction

A model that abstains when uncertain is performing selective prediction. Performance
is described by a risk–coverage curve: coverage is the fraction of questions
answered, and risk is the error rate among those answered. Lowering the confidence
threshold answers more questions at higher risk. The area under the risk–coverage
curve summarises the trade-off in a single number, where lower is better.

SQuAD 2.0 is useful for evaluating abstention because roughly one third of its
questions are deliberately unanswerable from the accompanying paragraph, providing
ground-truth labels for when a system should decline to answer.
