"""
Query Rewriter Prompts
======================
Prompt templates for query rewriting strategies.

Separated from query_rewriter.py to respect the ~200 LOC rule.
Reference: ADR-010 — Advanced Agentic RAG System
"""

# --- Strategy classifier ---------------------------------------------------

STRATEGY_CLASSIFIER_PROMPT = """You are a query strategy classifier for an Indian agricultural AI.

Given a user query, classify which rewrite strategy would improve retrieval:

- "step_back" → Specific query about a location/crop that benefits from broader context
- "hyde"      → Vague or short query where a hypothetical document improves recall
- "multi_query" → Multi-part or comparative query to decompose
- "none"      → Clear, well-formed query that needs no rewriting

Query: {query}

Respond with ONLY one word: step_back, hyde, multi_query, or none"""


# --- Step-back prompting ----------------------------------------------------

STEP_BACK_PROMPT = """You are an expert in Indian agriculture helping farmers.

Given a specific question, generate a broader "step-back" question that
would help retrieve more relevant background information.

Example:
  Specific: "How to control leaf curl on tomato in Kolar district?"
  Step-back: "What are common tomato viral diseases and their management?"

  Specific: "What is onion price in Hubli mandi today?"
  Step-back: "What are current onion market trends in Karnataka?"

Now generate ONE step-back question:
Specific: {query}
Step-back:"""


# --- HyDE (Hypothetical Document Embedding) --------------------------------

HYDE_PROMPT = """You are an expert in Indian agriculture.

Given a question, write a short hypothetical paragraph (3-5 sentences)
that would answer this question if it existed in a knowledge base.
The paragraph should be factual-sounding and specific to Indian agriculture.

Question: {query}

Hypothetical answer paragraph:"""


# --- Multi-query expansion --------------------------------------------------

MULTI_QUERY_PROMPT = """You are a helpful assistant generating search queries for
an Indian agricultural knowledge base.

Given the original question, generate 3 diverse reformulations that would
help retrieve all relevant information. Each reformulation should:
- Focus on a different aspect of the question
- Use different terminology
- Be self-contained

Original question: {query}

Return as a JSON array of 3 strings. Example:
["reformulation 1", "reformulation 2", "reformulation 3"]"""
