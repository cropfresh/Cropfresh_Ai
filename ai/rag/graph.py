"""
Agentic RAG Graph
==================
LangGraph-based agentic RAG workflow combining:
- Adaptive RAG (query routing)
- Corrective RAG (document grading + web fallback)  
- Self-RAG (hallucination checking)

This is the core workflow that orchestrates the entire RAG pipeline.
"""

from typing import Annotated, Any, Literal, Optional, TypedDict

from loguru import logger


class RAGState(TypedDict):
    """State for the agentic RAG workflow."""
    
    # Input
    question: str
    
    # Processing
    query_type: str  # vector, web, decompose, direct
    category: str
    sub_queries: list[str]
    
    # Retrieved content
    documents: list[Any]  # List of Document objects
    web_results: list[str]
    
    # Generation
    generation: str
    
    # Control flow
    web_search: str  # "Yes" or "No"
    retry_count: int
    max_retries: int
    
    # Output
    final_answer: str
    steps: list[str]


def create_rag_graph(
    knowledge_base,
    llm,
    web_search_tool=None,
):
    """
    Create the agentic RAG workflow graph.
    
    Args:
        knowledge_base: KnowledgeBase instance
        llm: LLM provider
        web_search_tool: Optional web search tool
        
    Returns:
        Compiled LangGraph workflow
    """
    from langgraph.graph import END, START, StateGraph
    
    from src.rag.grader import DocumentGrader, HallucinationChecker
    from src.rag.query_analyzer import QueryAnalyzer, QueryType
    from src.rag.retriever import RAGRetriever, decompose_query
    
    # Initialize components
    query_analyzer = QueryAnalyzer(llm=llm)
    retriever = RAGRetriever(knowledge_base=knowledge_base, llm=llm)
    grader = DocumentGrader(llm=llm)
    hallucination_checker = HallucinationChecker(llm=llm)
    
    # =========================================================================
    # NODE DEFINITIONS
    # =========================================================================
    
    async def analyze_query_node(state: RAGState) -> dict:
        """Analyze query and determine routing."""
        logger.info("---ANALYZE QUERY---")
        
        question = state["question"]
        steps = state.get("steps", [])
        steps.append("analyze_query")
        
        analysis = await query_analyzer.analyze(question)
        
        logger.info(f"Query type: {analysis.query_type.value}")
        logger.info(f"Category: {analysis.category.value}")
        
        return {
            "query_type": analysis.query_type.value,
            "category": analysis.category.value,
            "sub_queries": analysis.sub_queries,
            "steps": steps,
        }
    
    async def retrieve_node(state: RAGState) -> dict:
        """Retrieve documents from knowledge base."""
        logger.info("---RETRIEVE---")
        
        question = state["question"]
        category = state.get("category")
        sub_queries = state.get("sub_queries", [])
        steps = state.get("steps", [])
        steps.append("retrieve")
        
        # Handle decomposed queries
        if sub_queries and len(sub_queries) > 1:
            result = await retriever.retrieve_with_decomposition(
                query=question,
                sub_queries=sub_queries,
                top_k_per_query=3,
            )
        else:
            result = await retriever.retrieve(
                query=question,
                top_k=5,
                category=category if category != "general" else None,
            )
        
        logger.info(f"Retrieved {len(result.documents)} documents")
        
        return {
            "documents": result.documents,
            "steps": steps,
        }
    
    async def grade_documents_node(state: RAGState) -> dict:
        """Grade document relevance (CRAG)."""
        logger.info("---GRADE DOCUMENTS---")
        
        question = state["question"]
        documents = state.get("documents", [])
        steps = state.get("steps", [])
        steps.append("grade_documents")
        
        grading_result = await grader.grade_documents(documents, question)
        
        logger.info(f"Relevant: {len(grading_result.relevant_documents)}/{grading_result.total_graded}")
        logger.info(f"Web search needed: {grading_result.needs_web_search}")
        
        return {
            "documents": grading_result.relevant_documents,
            "web_search": "Yes" if grading_result.needs_web_search else "No",
            "steps": steps,
        }
    
    async def web_search_node(state: RAGState) -> dict:
        """Perform web search fallback."""
        logger.info("---WEB SEARCH---")
        
        question = state["question"]
        steps = state.get("steps", [])
        steps.append("web_search")
        
        web_results = []
        
        if web_search_tool:
            try:
                # Use provided web search tool
                results = await web_search_tool.search(question)
                web_results = results
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        else:
            logger.info("No web search tool configured - skipping")
        
        return {
            "web_results": web_results,
            "steps": steps,
        }
    
    async def generate_node(state: RAGState) -> dict:
        """Generate answer from retrieved context."""
        logger.info("---GENERATE---")
        
        from src.orchestrator.llm_provider import LLMMessage
        
        question = state["question"]
        documents = state.get("documents", [])
        web_results = state.get("web_results", [])
        steps = state.get("steps", [])
        steps.append("generate")
        
        # Build context
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}]\n{doc.text}")
        
        for i, result in enumerate(web_results, 1):
            context_parts.append(f"[Web Result {i}]\n{result}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant information found."
        
        # Generate with LLM
        system_prompt = """You are a helpful agricultural assistant for CropFresh AI.
Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so honestly.
Be concise but thorough. Use ₹ for prices when relevant."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm.generate(messages, temperature=0.7, max_tokens=500)
        generation = response.content
        
        logger.info(f"Generated answer: {generation[:100]}...")
        
        return {
            "generation": generation,
            "steps": steps,
        }
    
    async def check_hallucination_node(state: RAGState) -> dict:
        """Check for hallucinations (Self-RAG)."""
        logger.info("---CHECK HALLUCINATION---")
        
        generation = state.get("generation", "")
        documents = state.get("documents", [])
        question = state["question"]
        steps = state.get("steps", [])
        steps.append("check_hallucination")
        retry_count = state.get("retry_count", 0)
        
        if not documents:
            # No documents to check against
            return {
                "final_answer": generation,
                "steps": steps,
            }
        
        is_grounded, reasoning = await hallucination_checker.check(
            answer=generation,
            documents=documents,
            query=question,
        )
        
        if is_grounded:
            logger.info("Answer is grounded in documents")
            return {
                "final_answer": generation,
                "steps": steps,
            }
        else:
            logger.warning(f"Hallucination detected: {reasoning}")
            return {
                "retry_count": retry_count + 1,
                "steps": steps,
            }
    
    async def direct_answer_node(state: RAGState) -> dict:
        """Generate direct answer without retrieval."""
        logger.info("---DIRECT ANSWER---")
        
        from src.orchestrator.llm_provider import LLMMessage
        
        question = state["question"]
        steps = state.get("steps", [])
        steps.append("direct_answer")
        
        messages = [
            LLMMessage(
                role="system",
                content="You are CropFresh AI, an agricultural marketplace assistant. Be helpful and concise."
            ),
            LLMMessage(role="user", content=question),
        ]
        
        response = await llm.generate(messages, temperature=0.7, max_tokens=300)
        
        return {
            "final_answer": response.content,
            "steps": steps,
        }
    
    # =========================================================================
    # ROUTING FUNCTIONS
    # =========================================================================
    
    def route_query(state: RAGState) -> Literal["retrieve", "web_search", "direct_answer"]:
        """Route based on query type."""
        query_type = state.get("query_type", "vector")
        
        if query_type == "direct":
            return "direct_answer"
        elif query_type == "web":
            return "web_search"
        else:  # vector or decompose
            return "retrieve"
    
    def decide_to_search(state: RAGState) -> Literal["web_search", "generate"]:
        """Decide whether to do web search based on grading."""
        web_search = state.get("web_search", "No")
        
        if web_search == "Yes":
            return "web_search"
        return "generate"
    
    def decide_to_retry(state: RAGState) -> Literal["retrieve", "end"]:
        """Decide whether to retry after hallucination detection."""
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        final_answer = state.get("final_answer")
        
        if final_answer:
            return "end"
        elif retry_count < max_retries:
            return "retrieve"
        else:
            return "end"
    
    # =========================================================================
    # BUILD GRAPH
    # =========================================================================
    
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_hallucination", check_hallucination_node)
    workflow.add_node("direct_answer", direct_answer_node)
    
    # Add edges
    workflow.add_edge(START, "analyze_query")
    
    # Route from analysis
    workflow.add_conditional_edges(
        "analyze_query",
        route_query,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "direct_answer": "direct_answer",
        }
    )
    
    # After retrieval, grade documents
    workflow.add_edge("retrieve", "grade_documents")
    
    # After grading, decide web search or generate
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_search,
        {
            "web_search": "web_search",
            "generate": "generate",
        }
    )
    
    # After web search, generate
    workflow.add_edge("web_search", "generate")
    
    # After generation, check hallucination
    workflow.add_edge("generate", "check_hallucination")
    
    # After hallucination check, retry or end
    workflow.add_conditional_edges(
        "check_hallucination",
        decide_to_retry,
        {
            "retrieve": "retrieve",
            "end": END,
        }
    )
    
    # Direct answer goes to end
    workflow.add_edge("direct_answer", END)
    
    # Compile and return
    graph = workflow.compile()
    
    return graph


async def run_agentic_rag(
    question: str,
    knowledge_base,
    llm,
    web_search_tool=None,
) -> dict:
    """
    Run the agentic RAG pipeline.
    
    Args:
        question: User question
        knowledge_base: KnowledgeBase instance
        llm: LLM provider
        web_search_tool: Optional web search tool
        
    Returns:
        Final state with answer
    """
    graph = create_rag_graph(knowledge_base, llm, web_search_tool)
    
    initial_state = {
        "question": question,
        "query_type": "",
        "category": "",
        "sub_queries": [],
        "documents": [],
        "web_results": [],
        "generation": "",
        "web_search": "No",
        "retry_count": 0,
        "max_retries": 2,
        "final_answer": "",
        "steps": [],
    }
    
    result = await graph.ainvoke(initial_state)
    
    return result
