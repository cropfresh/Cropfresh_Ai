"""
Comprehensive Test Script for Phases 11-15
==========================================
Tests all 25 components across 5 phases.
"""

import asyncio
from datetime import datetime

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def header(text: str):
    print(f"\n{BLUE}{'='*60}")
    print(f" {text}")
    print(f"{'='*60}{RESET}\n")

def success(text: str):
    print(f"{GREEN}✓ {text}{RESET}")

def error(text: str):
    print(f"{RED}✗ {text}{RESET}")

def info(text: str):
    print(f"{YELLOW}  → {text}{RESET}")


async def test_phase_11_research():
    """Test Phase 11: Deep Research Agent"""
    header("PHASE 11: Deep Research Agent")
    
    # Test 1: Models
    try:
        from src.agents.research.models import (
            Citation, Finding, ResearchStep, ResearchPlan, 
            ResearchReport, SourceType, StepStatus
        )
        
        # Create a citation
        citation = Citation(
            title="Tomato Prices in Karnataka",
            url="https://agmarknet.gov.in",
            source_type=SourceType.WEB,
        )
        success(f"Citation: {citation.to_apa()[:50]}...")
        
        # Create a finding
        finding = Finding(
            content="Tomato prices are ₹45/kg in Bengaluru",
            summary="Current tomato price",
            source_type=SourceType.WEB,
            reliability_score=0.8,
            citation=citation,
        )
        success(f"Finding: {finding.summary}")
        
        # Create a research plan
        step = ResearchStep(
            step_id="1",  # Must be string
            question="What are current tomato prices?",
            source_types=[SourceType.WEB, SourceType.KNOWLEDGE_BASE],
        )
        plan = ResearchPlan(
            objective="Research tomato prices",
            steps=[step],
        )
        success(f"ResearchPlan: {len(plan.steps)} steps, progress: {plan.progress:.0%}")
        
        # Create a report
        report = ResearchReport(
            title="Tomato Price Report",
            query="tomato prices",
            summary="Analysis of current tomato prices",
            findings=[finding],
            citations=[citation],
        )
        success(f"ResearchReport: {report.title}")
        info(f"Report markdown: {len(report.to_markdown())} chars")
        
    except Exception as e:
        error(f"Models: {e}")
        return False
    
    # Test 2: Research Planner
    try:
        from src.agents.research.planner import ResearchPlanner
        
        planner = ResearchPlanner()  # No LLM for simple test
        plan = await planner.create_plan("Best tomato varieties for summer")
        success(f"ResearchPlanner: Generated {len(plan.steps)} steps")
        for step in plan.steps[:2]:
            info(f"Step {step.step_id}: {step.question[:40]}...")
        
    except Exception as e:
        error(f"Planner: {e}")
        return False
    
    # Test 3: Source Discovery
    try:
        from src.agents.research.source_discovery import SourceDiscovery
        
        discovery = SourceDiscovery()
        # Just test URL building (no actual web requests)
        urls = discovery._build_search_urls("tomato price Karnataka")
        success(f"SourceDiscovery: Built {len(urls)} search URLs")
        info(f"URLs: {urls[:2]}")
        
    except Exception as e:
        error(f"SourceDiscovery: {e}")
        return False
    
    # Test 4: Verifier
    try:
        from src.agents.research.verifier import SourceVerifier
        
        verifier = SourceVerifier()
        test_finding = Finding(
            content="Test content with data: ₹45/kg in 2026",
            source_url="https://agmarknet.gov.in/prices",
            source_type=SourceType.WEB,
        )
        verified = await verifier.verify(test_finding)
        success(f"SourceVerifier: Reliability score = {verified.reliability_score:.2f}")
        
    except Exception as e:
        error(f"Verifier: {e}")
        return False
    
    # Test 5: Citation Manager
    try:
        from src.agents.research.citation_manager import CitationManager
        
        manager = CitationManager()
        ref = manager.add_citation(Citation(
            title="Agricultural Market Report",
            url="https://example.com",
            source_type=SourceType.WEB,
        ))
        success(f"CitationManager: Added citation {ref}")
        info(f"Bibliography:\n{manager.get_bibliography()}")
        
    except Exception as e:
        error(f"CitationManager: {e}")
        return False
    
    # Test 6: Research Memory
    try:
        from src.agents.research.memory import ResearchMemory
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        memory = ResearchMemory(memory_dir=temp_dir / "test_memory")
        
        entry_id = await memory.store(
            query="tomato prices",
            findings=[finding],
        )
        success(f"ResearchMemory: Stored entry {entry_id}")
        
        similar = await memory.find_similar("tomato market rates")
        info(f"Found {len(similar)} similar entries")
        
    except Exception as e:
        error(f"ResearchMemory: {e}")
        return False
    
    # Test 7: Research Agent
    try:
        from src.agents.research.research_agent import ResearchAgent
        
        agent = ResearchAgent()
        success("ResearchAgent: Initialized successfully")
        info("Full research requires LLM - initialization verified")
        
    except Exception as e:
        error(f"ResearchAgent: {e}")
        return False
    
    print(f"\n{GREEN}Phase 11: All 7 components passed!{RESET}")
    return True


async def test_phase_12_resilience():
    """Test Phase 12: Multi-Agent Self-Healing"""
    header("PHASE 12: Multi-Agent Self-Healing")
    
    # Test 1: Reflection
    try:
        from src.resilience.reflection import ReflectionEngine, IssueType
        
        reflector = ReflectionEngine()  # No LLM for simple test
        result = await reflector.reflect(
            query="What is the price?",
            response="The price is probably around something."
        )
        success(f"ReflectionEngine: Found {len(result.issues_found)} issues")
        info(f"Confidence: {result.confidence_before:.2f} -> {result.confidence_after:.2f}")
        
    except Exception as e:
        error(f"Reflection: {e}")
        return False
    
    # Test 2: Error Recovery
    try:
        from src.resilience.recovery import ErrorRecovery, RetryPolicy
        
        recovery = ErrorRecovery(default_policy=RetryPolicy(max_retries=2))
        
        # Test with a successful function
        async def succeed():
            return "success!"
        
        result = await recovery.execute_with_retry(succeed)
        success(f"ErrorRecovery: {result.result} in {result.attempts} attempt(s)")
        
        # Test delay calculation
        policy = RetryPolicy(initial_delay_sec=1.0)
        delays = [policy.get_delay(i) for i in range(3)]
        info(f"Backoff delays: {[f'{d:.1f}s' for d in delays]}")
        
    except Exception as e:
        error(f"Recovery: {e}")
        return False
    
    # Test 3: Circuit Breaker
    try:
        from src.resilience.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker("test_service")
        
        # Test successful calls
        async def ok_func():
            return "ok"
        
        result = await breaker.call(ok_func)
        success(f"CircuitBreaker: State = {breaker.state.value}, Result = {result}")
        info(f"Stats: {breaker.stats.total_requests} requests, {breaker.stats.failure_count} failures")
        
    except Exception as e:
        error(f"CircuitBreaker: {e}")
        return False
    
    # Test 4: Health Monitor
    try:
        from src.resilience.health_monitor import HealthMonitor, HealthStatus
        
        monitor = HealthMonitor()
        
        # Record some requests
        monitor.record_request("test_agent", latency_ms=150)
        monitor.record_request("test_agent", latency_ms=200)
        monitor.record_error("test_agent", "Test error")
        
        health = monitor.get_health("test_agent")
        success(f"HealthMonitor: Status = {health.status.value}")
        info(f"Requests: {health.request_count}, Errors: {health.error_count}")
        info(f"Avg latency: {health.avg_latency_ms:.0f}ms, Error rate: {health.error_rate:.0%}")
        
    except Exception as e:
        error(f"HealthMonitor: {e}")
        return False
    
    # Test 5: Task Decomposer
    try:
        from src.resilience.task_decomposer import TaskDecomposer, TaskStatus
        
        decomposer = TaskDecomposer()  # No LLM
        graph = await decomposer.decompose("Research and recommend best crop")
        
        success(f"TaskDecomposer: Created graph with {len(graph.tasks)} tasks")
        for task in graph.tasks[:2]:
            info(f"Task: {task.name} (executor: {task.executor})")
        
        ready = graph.get_ready_tasks()
        info(f"Ready to execute: {len(ready)} tasks")
        
    except Exception as e:
        error(f"TaskDecomposer: {e}")
        return False
    
    # Test 6: Feedback Loop
    try:
        from src.resilience.feedback import FeedbackLoop
        
        loop = FeedbackLoop()
        
        # Record feedback
        loop.record_feedback(
            query="What is tomato price?",
            response="₹45/kg",
            agent_name="commerce_agent",
            was_helpful=True,
        )
        loop.record_feedback(
            query="tomato pest control",
            response="Use neem oil",
            agent_name="agronomy_agent",
            was_helpful=False,
            error_type="incomplete",
        )
        
        stats = loop.get_stats()
        success(f"FeedbackLoop: {stats.total_feedback} feedback entries")
        info(f"Positive: {stats.positive_count}, Negative: {stats.negative_count}")
        
        suggestions = loop.get_improvement_suggestions()
        if suggestions:
            info(f"Suggestions: {suggestions[0][:50]}...")
        
    except Exception as e:
        error(f"FeedbackLoop: {e}")
        return False
    
    print(f"\n{GREEN}Phase 12: All 6 components passed!{RESET}")
    return True


async def test_phase_13_enhanced_rag():
    """Test Phase 13: Enhanced LLM-RAG"""
    header("PHASE 13: Enhanced LLM-RAG")
    
    # Test 1: Instructed Retriever
    try:
        from src.rag.enhanced.instructed_retriever import InstructedRetriever
        
        retriever = InstructedRetriever(enable_instruction_generation=False)
        
        # Test instruction parsing
        keywords = retriever._extract_keywords("What is the price of tomatoes in Karnataka?")
        success(f"InstructedRetriever: Extracted keywords {keywords}")
        
    except Exception as e:
        error(f"InstructedRetriever: {e}")
        return False
    
    # Test 2: Strategy Selector
    try:
        from src.rag.enhanced.strategy_selector import StrategySelector, RetrievalStrategy
        
        selector = StrategySelector()
        
        # Test different queries
        queries = [
            "What is the relationship between rainfall and crop yield?",
            "Define photosynthesis",
            "Compare organic vs chemical fertilizers",
        ]
        
        success("StrategySelector: Testing queries...")
        for query in queries:
            selection = await selector.select(query)
            info(f"'{query[:30]}...' → {selection.primary_strategy.value} (conf: {selection.confidence:.2f})")
        
    except Exception as e:
        error(f"StrategySelector: {e}")
        return False
    
    # Test 3: Bidirectional RAG
    try:
        from src.rag.enhanced.bidirectional_rag import BidirectionalRAG
        
        brag = BidirectionalRAG()
        
        # Test concept extraction
        concepts = brag._extract_related_concepts("Tomato cultivation requires good irrigation and fertilizer")
        success(f"BidirectionalRAG: Extracted concepts {concepts}")
        
    except Exception as e:
        error(f"BidirectionalRAG: {e}")
        return False
    
    # Test 4: Prompt Optimizer
    try:
        from src.rag.enhanced.prompt_optimizer import PromptOptimizer
        
        optimizer = PromptOptimizer()
        result = await optimizer.optimize(
            query="What is the best tomato variety for Karnataka?",
            context=["Arka Rakshak is a popular variety", "Karnataka has tropical climate"],
            agent_type="agronomy",
            task_type="recommendation",
        )
        
        success(f"PromptOptimizer: Generated ~{result.estimated_tokens} tokens")
        info(f"System prompt: {result.system_prompt[:50]}...")
        info(f"Examples included: {len(result.examples)}")
        
    except Exception as e:
        error(f"PromptOptimizer: {e}")
        return False
    
    print(f"\n{GREEN}Phase 13: All 4 components passed!{RESET}")
    return True


async def test_phase_14_autonomous():
    """Test Phase 14: Autonomous Task Completion"""
    header("PHASE 14: Autonomous Task Completion")
    
    # Test 1: Goal Agent
    try:
        from src.autonomous.goal_agent import GoalAgent, Objective, ObjectiveTree
        
        agent = GoalAgent()  # No LLM
        tree = await agent.create_objective_tree("Research and recommend best crop")
        
        success(f"GoalAgent: Created tree with {len(tree.objectives)} objectives")
        info(f"Root: {tree.objectives.get(tree.root_objective).name}")
        info(f"Progress: {tree.progress:.0%}")
        
        ready = tree.get_ready_objectives()
        info(f"Ready objectives: {len(ready)}")
        
    except Exception as e:
        error(f"GoalAgent: {e}")
        return False
    
    # Test 2: PEAR Loop
    try:
        from src.autonomous.pear_loop import PEARLoop, PlanStep
        
        pear = PEARLoop()  # No LLM
        
        # Test planning
        plan = await pear._plan("Find tomato prices", "", None)
        success(f"PEARLoop: Generated {len(plan)} plan steps")
        for step in plan[:2]:
            info(f"Step {step.step_id}: {step.action[:40]}...")
        
    except Exception as e:
        error(f"PEARLoop: {e}")
        return False
    
    # Test 3: Progress Monitor
    try:
        from src.autonomous.progress_monitor import ProgressMonitor
        
        monitor = ProgressMonitor()
        
        # Create and track a task
        task = monitor.create_task("test_001", "Research Task", total_steps=5)
        monitor.update_progress("test_001", step=2, current="Gathering data")
        
        progress = monitor.get_progress("test_001")
        success(f"ProgressMonitor: Task at {progress.progress_percent:.0f}%")
        info(f"Current: {progress.current_step}")
        info(f"ETA: {progress.eta_seconds:.0f}s" if progress.eta_seconds else "ETA: calculating...")
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        info(f"Dashboard: {dashboard['running']} running, {dashboard['completed']} completed")
        
    except Exception as e:
        error(f"ProgressMonitor: {e}")
        return False
    
    # Test 4: Task Persistence
    try:
        from src.autonomous.persistence import TaskPersistence
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        persistence = TaskPersistence(tasks_dir=temp_dir / "test_tasks")
        
        # Save a task
        saved = await persistence.save("task_123", {
            "objective": "Test task",
            "progress": 50,
            "status": "running",
        })
        success(f"TaskPersistence: Saved task = {saved}")
        
        # Load it back
        loaded = await persistence.load("task_123")
        info(f"Loaded: {loaded}")
        
        # List tasks
        tasks = await persistence.list_tasks()
        info(f"Total tasks: {len(tasks)}")
        
    except Exception as e:
        error(f"TaskPersistence: {e}")
        return False
    
    print(f"\n{GREEN}Phase 14: All 4 components passed!{RESET}")
    return True


async def test_phase_15_production():
    """Test Phase 15: Production Readiness"""
    header("PHASE 15: Production Readiness")
    
    # Test 1: Observability
    try:
        from src.production.observability import (
            setup_observability, trace_agent, get_metrics, 
            record_tokens, AgentMetrics
        )
        
        # Test metrics
        record_tokens("test_agent", tokens_in=100, tokens_out=50)
        metrics = get_metrics("test_agent")
        success(f"Observability: Recorded tokens")
        info(f"Metrics: {metrics}")
        
        # Test decorator (without actual execution)
        @trace_agent("decorated_agent")
        async def test_func():
            return "traced!"
        
        result = await test_func()
        success(f"Trace decorator: {result}")
        
    except Exception as e:
        error(f"Observability: {e}")
        return False
    
    # Test 2: Rate Limiter
    try:
        from src.production.rate_limiter import RateLimiter, RateLimitExceeded
        
        limiter = RateLimiter()
        
        # Should succeed
        await limiter.check("user_1")
        await limiter.check("user_1")
        await limiter.check("user_1")
        
        remaining = limiter.get_remaining("user_1")
        success(f"RateLimiter: 3 requests allowed")
        info(f"Remaining: minute={remaining['minute']['remaining']}, burst={remaining['burst']['available']:.1f}")
        
    except Exception as e:
        error(f"RateLimiter: {e}")
        return False
    
    # Test 3: Response Cache
    try:
        from src.production.cache import ResponseCache
        
        cache = ResponseCache()
        
        # Set and get
        await cache.set("test_key", {"data": "cached value"}, ttl=60)
        value = await cache.get("test_key")
        success(f"ResponseCache: Cached and retrieved")
        info(f"Value: {value}")
        
        # Check stats
        stats = cache.get_stats()
        info(f"Stats: {stats.total_hits} hits, {stats.total_misses} misses, {stats.hit_rate:.0%} hit rate")
        
        # Generate key
        key = ResponseCache.generate_key("query", user_id="123", limit=10)
        info(f"Generated key: {key}")
        
    except Exception as e:
        error(f"ResponseCache: {e}")
        return False
    
    # Test 4: Production Config
    try:
        from src.production.config import ProductionConfig, load_config, Environment
        
        # Test default config
        config = ProductionConfig()
        success(f"ProductionConfig: environment={config.environment.value}")
        info(f"LLM: {config.llm.provider}/{config.llm.model}")
        info(f"Features: research={config.features.enable_research_agent}, scraping={config.features.enable_web_scraping}")
        
        # Test load from env
        loaded = load_config()
        info(f"Loaded config for {loaded.environment.value}")
        
    except Exception as e:
        error(f"ProductionConfig: {e}")
        return False
    
    print(f"\n{GREEN}Phase 15: All 4 components passed!{RESET}")
    return True


async def main():
    """Run all tests"""
    print(f"\n{BLUE}{'#'*60}")
    print(f"#  PHASES 11-15 COMPREHENSIVE TEST SUITE")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}{RESET}")
    
    results = {}
    
    # Run each phase
    results["Phase 11"] = await test_phase_11_research()
    results["Phase 12"] = await test_phase_12_resilience()
    results["Phase 13"] = await test_phase_13_enhanced_rag()
    results["Phase 14"] = await test_phase_14_autonomous()
    results["Phase 15"] = await test_phase_15_production()
    
    # Summary
    header("FINAL SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for phase, passed_flag in results.items():
        status = f"{GREEN}PASSED{RESET}" if passed_flag else f"{RED}FAILED{RESET}"
        print(f"  {phase}: {status}")
    
    print(f"\n{'='*40}")
    if passed == total:
        print(f"{GREEN}🎉 ALL {total} PHASES PASSED!{RESET}")
    else:
        print(f"{YELLOW}⚠️  {passed}/{total} phases passed{RESET}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    asyncio.run(main())
