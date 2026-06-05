import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME="gemma-4-31b-it"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

ANSWER_RELEVANCY_PROMPT = """You are an evaluator for question-answering systems.

Your task is to determine how relevant the ANSWER is to the QUESTION.

Evaluate whether the ANSWER directly addresses the user's intent and information need.

Use ONLY the QUESTION and ANSWER.
Do NOT evaluate factual correctness unless it affects relevance.

Evaluation Definition

Answer relevancy measures how well the ANSWER addresses the QUESTION.

A highly relevant answer:

Directly answers the question
Addresses the main user intent
Stays focused on the requested topic
Provides information the user actually asked for
Avoids unnecessary tangents or unrelated details

A low relevance answer:

Avoids the question
Changes the subject
Contains mostly unrelated information
Is overly vague or generic
Answers only a small portion of the question
Gives boilerplate text instead of a real answer

Instructions

Identify the primary intent of the QUESTION.

Determine whether the ANSWER addresses that intent directly.

Evaluate:
Topical relevance
Completeness relative to the question
Presence of irrelevant information
Whether the answer avoids or evades the question

Do NOT penalize:
Writing style
Grammar
Tone

Do NOT evaluate factual accuracy unless the answer is so incorrect that it no longer meaningfully answers the question.

Be strict about partial answers.

An answer that addresses only one part of a multi-part question should receive a reduced score.

Special Cases

If the ANSWER is completely unrelated to the QUESTION:
score = 0.0

If the ANSWER is generic and could apply to almost any question:
score should be low.

If the ANSWER contains both relevant and irrelevant information:
reduce the score proportionally.

If the ANSWER refuses to answer or evades the question:
score should reflect how much useful information remains.

Scoring Rules

Return a score between 0.0 and 1.0.

Use this scale:

1.0:
Fully addresses the user's intent with focused and complete information.

0.7–0.9:
Mostly relevant, but missing minor details or containing small tangents.

0.4–0.6:
Partially relevant; answers only part of the question or includes substantial irrelevant content.

0.1–0.3:
Weakly related to the question but largely unhelpful.

0.0:
Completely irrelevant or non-responsive.

Required Output Format

Return ONLY valid JSON.

{{
"question_intent": "...",
"coverage": {{
"addressed": [
"..."
],
"missing": [
"..."
]
}},
"irrelevant_content": [
"..."
],
"score": 0.0,
"reason": "Short explanation"
}}

QUESTION

{question}

ANSWER

{answer}"""


FAITHFULNESS_PROMPT = """
You are an evaluator for retrieval-augmented generation (RAG) systems.

Your task is to determine whether the ANSWER is fully supported by the provided CONTEXT.

Use ONLY the information in the CONTEXT.
Do NOT use prior knowledge, assumptions, or external facts.

Evaluation Definition

Faithfulness measures whether the factual claims in the ANSWER are grounded in the CONTEXT.

An answer is faithful if:

Every factual claim is supported by the context
No unsupported facts are introduced
No contradictions exist
No fabricated details, numbers, entities, events, or relationships are added

An answer is NOT fully faithful if:

It contains hallucinated information
It adds details absent from the context
It overstates certainty
It makes unsupported causal claims or implications
It contradicts the context

Instructions

Extract all factual claims from the ANSWER.

For each claim:
Determine whether it is:
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
CONTRADICTED

Ignore:
Writing quality
Grammar
Style
Usefulness

Do NOT reward plausibility.

A claim may sound correct but must still be marked unsupported if absent from the context.

Be strict.

Missing evidence means unsupported.

Scoring Rules

Return a faithfulness score between 0.0 and 1.0.

Use this scale:

1.0:
Every factual claim is directly supported by the context.

0.7–0.9:
Mostly supported, but contains minor unsupported embellishments or weak inferences.

0.4–0.6:
Mixture of supported and unsupported claims.

0.1–0.3:
Major hallucinations or unsupported assertions.

0.0:
The answer is entirely unsupported or contradicted by the context.

If any major factual claim is contradicted by the context, the score should be heavily reduced.

Required Output Format

Return ONLY valid JSON.

{{
"claims": [
{{
"claim": "...",
"label": "SUPPORTED"
}}
],
"score": 0.0,
"reason": "Short explanation"
}}

CONTEXT

{context}

ANSWER

{answer}
"""

def _call_gemini(prompt: str) -> str:
   

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text


def _clean_json_response(text: str) -> str:

    cleaned = text.strip()

    cleaned = cleaned.replace(
        "```json",
        ""
    )

    cleaned = cleaned.replace(
        "```",
        ""
    )

    return cleaned.strip()


def _extract_score(response_text: str) -> float:
   

    try:
        cleaned = _clean_json_response(
            response_text
        )

        parsed = json.loads(cleaned)

        score = float(
            parsed.get("score", 0.0)
        )

        # clamp into [0,1]
        score = max(0.0, min(score, 1.0))

        return score

    except Exception:
        return 0.0


def compute(
    question: str,
    answer: str,
    context: str,
) -> dict:
    """
    Compute:
    - faithfulness
    - answer relevancy
    """

    faithfulness_prompt = (
        FAITHFULNESS_PROMPT.format(
            context=context,
            answer=answer,
        )
    )

    answer_relevancy_prompt = (
        ANSWER_RELEVANCY_PROMPT.format(
            question=question,
            answer=answer,
        )
    )

    # raw Gemini responses
    faithfulness_response = _call_gemini(
        faithfulness_prompt
    )

    answer_relevancy_response = _call_gemini(
        answer_relevancy_prompt
    )

    # parsed scores
    faithfulness_score = _extract_score(
        faithfulness_response
    )

    answer_relevancy_score = _extract_score(
        answer_relevancy_response
    )

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": answer_relevancy_score,
    }