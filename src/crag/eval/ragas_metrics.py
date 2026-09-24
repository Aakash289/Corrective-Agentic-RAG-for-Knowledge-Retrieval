"""
src/crag/eval/ragas_metrics.py

Wraps the ragas library's evaluation, using Claude as the judge LLM
instead of ragas's OpenAI default, and a local sentence-transformers
model for embeddings (AnswerRelevancy needs an embedding model to
compare a generated question back against the original, it doesn't use
the judge LLM for that part). Same embedding model embed_index.py
already uses, no new dependency for that half.

This is a genuinely different thing from score_response_node's inline
scoring in the graph itself: that runs per-request with a simplified,
partly boolean-derived faithfulness_score. This runs offline, once,
over a fixed evaluation set, using ragas's real metric implementations,
each of which does its own internal LLM-judged decomposition (breaking
an answer into individual claims and checking each one, for
faithfulness, rather than one holistic yes/no judgment the way
output_guardrail.py's check_output does).

The ragas API shown here matches the library's current documented usage
as of this file being written. Ragas has changed its API meaningfully
across versions before (SingleTurnSample/EvaluationDataset is a newer
shape than the plain Dataset-based one some older tutorials show), so
treat the first real run of this file as a verification step, not an
assumption, the same way earlier surprises in this project (the
Anthropic SDK dropping temperature as a parameter) turned out to be
real, current API behavior rather than something to code around blind.
"""
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

JUDGE_MODEL = "claude-sonnet-4-6"  # NOT claude-sonnet-5: confirmed directly,
                                      # every one of 32 real judge calls failed
                                      # with "temperature is deprecated for
                                      # this model" when this was claude-sonnet-5.
                                      # ragas's own metric implementations
                                      # explicitly pass a temperature value at
                                      # call time when invoking the judge LLM,
                                      # this isn't fixable from our side, it's
                                      # ragas's own code sending it, not
                                      # ChatAnthropic's default (which is None
                                      # unless overridden). A judge doesn't
                                      # need to be the newest model, consistent
                                      # grading behavior matters more here than
                                      # being on the latest release, so this
                                      # sidesteps the incompatibility entirely
                                      # rather than patching ragas's internals.
                                      # If this exact model string turns out to
                                      # be wrong (API model names sometimes
                                      # need a date suffix), that's a spelling
                                      # fix, not a sign this approach is wrong.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # matches embed_index.py's model, kept
                                        # consistent rather than introducing
                                        # a second embedding model into the
                                        # project just for evaluation

_evaluator_llm = None
_evaluator_embeddings = None


def _get_evaluator_llm():
    # No return type annotation: Pylance flagged LangchainLLMWrapper as
    # "Variable not allowed in type expression", the installed ragas
    # version's actual object isn't a proper type-checkable class the
    # way the docs implied, real API drift, same pattern as the
    # temperature issue. This is a static-analysis-only concern, not a
    # runtime one, so the annotation is just dropped rather than fought.
    global _evaluator_llm
    if _evaluator_llm is None:
        _evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL))
    return _evaluator_llm


def _get_evaluator_embeddings():
    global _evaluator_embeddings
    if _evaluator_embeddings is None:
        _evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    return _evaluator_embeddings


def build_sample(question: str, answer: str, retrieved_contexts: list[str], reference: str | None = None) -> SingleTurnSample:
    """One evaluation row: the question asked, the answer the graph
    actually produced, the raw text of every chunk that was retrieved
    for it (not just the ones cited, ragas's context metrics want the
    full retrieved set to judge precision/recall against), and
    optionally a reference answer the eval set author considers
    correct, needed for ContextRecall and ContextPrecision. reference
    is optional (defaults to None, sent to ragas as an empty string)
    for ad-hoc questions asked live in the UI, which have no
    pre-written ground truth, in that case only run_ragas_evaluation's
    include_reference_metrics=False mode should be used, the two
    metrics that need a reference can't be computed meaningfully
    without one."""
    return SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts,
        reference=reference or "",
    )


def run_ragas_evaluation(samples: list[SingleTurnSample], include_reference_metrics: bool = True):
    """Runs ragas metrics over the sample set, returns ragas's own
    Result object, which behaves like a dict of aggregate scores and
    also converts to a per-row DataFrame via .to_pandas().

    include_reference_metrics controls whether ContextPrecision and
    ContextRecall run: both need a real reference (ground-truth) answer
    to judge against, meaningless without one. Set to False for ad-hoc
    questions with no pre-written reference, in which case only
    Faithfulness and AnswerRelevancy run, both of which score directly
    from the response and retrieved context, no reference needed."""
    dataset = EvaluationDataset(samples=samples)
    llm = _get_evaluator_llm()
    embeddings = _get_evaluator_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]
    if include_reference_metrics:
        metrics.append(ContextPrecision(llm=llm))
        metrics.append(ContextRecall(llm=llm))

    return evaluate(dataset=dataset, metrics=metrics)