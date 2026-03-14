"""
Query Processor Prompts
"""

HYDE_PROMPT = """You are an agricultural expert writing a detailed answer.
Write a comprehensive paragraph that would answer this question about farming in India.
Include specific details, numbers, and practical recommendations.
Do not mention that this is hypothetical.

Question: {query}

Answer:"""

MULTI_QUERY_PROMPT = """You are an AI assistant helping farmers find information.
Generate {count} different search queries that would help find information for this user question.
Each query should approach the topic from a different angle or focus on different aspects.
Make queries specific and searchable.

Original question: {query}

Generate {count} alternative search queries, one per line:"""

STEP_BACK_PROMPT = """You are an AI assistant that reformulates specific questions into more general ones.
Given a specific question about agriculture, generate a broader, more abstract question 
that could provide background knowledge helpful for answering the original.

Original question: {query}

Step-back question (more general):"""

DECOMPOSE_PROMPT = """You are an AI assistant that breaks down complex questions.
If the question is simple, just output the original question.
If complex, break it into 2-3 simpler sub-questions that together answer the original.

Original question: {query}

Sub-questions (one per line, or just the original if simple):"""

REWRITE_PROMPT = """You are an AI assistant optimizing search queries.
Rewrite this question to be more effective for searching an agricultural knowledge base.
Remove filler words, add relevant agricultural terms, and make it more specific.

Original: {query}

Optimized search query:"""
