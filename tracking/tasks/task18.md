# Task 18: Build RAGAS Evaluation Framework with Golden Dataset

> **Priority:** 🟠 P1 | **Phase:** 5 | **Effort:** 3–4 days  
> **Files:** `src/evaluation/` [NEW directory]  
> **Score Target:** 9/10 — Systematic RAG evaluation with baseline metrics

---

## 📌 Problem Statement

No evaluation baseline exists. Need to build a golden dataset of Q&A pairs and evaluate the RAG pipeline using RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall).

---

## 🔬 Research Findings

### RAGAS Metrics (2025)
| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Are answers grounded in retrieved context? (anti-hallucination) | > 0.80 |
| **Answer Relevancy** | Does answer directly address the question? | > 0.75 |
| **Context Precision** | Is retrieved context relevant (signal-to-noise)? | > 0.70 |
| **Context Recall** | Does context contain all needed info? | > 0.70 |

### Golden Dataset Structure
```json
{
    "question": "What is the current price of tomato in Bangalore?",
    "ground_truth": "Tomato price in Bangalore is approximately ₹25-30/kg based on the latest Agmarknet data.",
    "contexts": ["Tomato prices in Karnataka mandis..."],
    "agent_domain": "commerce",
    "difficulty": "easy",
    "language": "en"
}
```

---

## 🏗️ Implementation Spec

### Directory Structure
```
src/evaluation/
├── __init__.py
├── golden_dataset.json          # 50+ Q&A pairs
├── ragas_evaluator.py           # RAGAS metrics calculator
├── eval_runner.py               # Batch evaluation script
├── report_generator.py          # Generates evaluation report
└── datasets/
    ├── agronomy_qa.json         # 15 agronomy Q&A pairs
    ├── commerce_qa.json         # 15 commerce/pricing Q&A pairs
    ├── platform_qa.json         # 10 platform Q&A pairs
    └── multilingual_qa.json     # 10 multilingual Q&A pairs
```

### Golden Dataset Categories
| Category | Count | Example Question |
|----------|-------|-----------------|
| Agronomy | 15 | "How to prevent tomato blight in Karnataka?" |
| Commerce | 15 | "Should I sell my onions now or wait?" |
| Platform | 10 | "How do I register as a farmer on CropFresh?" |
| Multilingual | 10 | "ಟೊಮೆಟೊ ಬೆಲೆ ಎಷ್ಟು?" (Kannada) |

### Evaluator Implementation
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

class RAGASEvaluator:
    """
    Evaluate CropFresh RAG pipeline against golden dataset.
    
    Usage:
        evaluator = RAGASEvaluator()
        results = await evaluator.run_evaluation("golden_dataset.json")
        evaluator.generate_report(results, "eval_report.md")
    """
    
    async def run_evaluation(self, dataset_path: str) -> EvalResults:
        """Run RAGAS evaluation on the full golden dataset."""
        dataset = self._load_dataset(dataset_path)
        
        # For each Q&A pair, get RAG pipeline response
        eval_data = []
        for item in dataset:
            response = await self.rag_pipeline.query(item["question"])
            eval_data.append({
                "question": item["question"],
                "answer": response.answer,
                "contexts": response.contexts,
                "ground_truth": item["ground_truth"],
            })
        
        # Run RAGAS evaluation
        results = evaluate(
            Dataset.from_list(eval_data),
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        
        return EvalResults(
            faithfulness=results['faithfulness'],
            answer_relevancy=results['answer_relevancy'],
            context_precision=results['context_precision'],
            context_recall=results['context_recall'],
            per_question=results.to_pandas(),
        )
    
    def generate_report(self, results: EvalResults, output_path: str):
        """Generate markdown evaluation report."""
        report = f"""
# CropFresh RAG Evaluation Report
**Date:** {datetime.now().isoformat()}
**Dataset:** {len(results.per_question)} questions

## Overall Metrics
| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Faithfulness | {results.faithfulness:.3f} | > 0.80 | {'✅' if results.faithfulness > 0.8 else '❌'} |
| Answer Relevancy | {results.answer_relevancy:.3f} | > 0.75 | {'✅' if results.answer_relevancy > 0.75 else '❌'} |
| Context Precision | {results.context_precision:.3f} | > 0.70 | {'✅' if results.context_precision > 0.7 else '❌'} |
| Context Recall | {results.context_recall:.3f} | > 0.70 | {'✅' if results.context_recall > 0.7 else '❌'} |

## Worst Performing Questions
{self._format_worst_questions(results)}
"""
        Path(output_path).write_text(report)
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Golden dataset with 50+ Q&A pairs across 4 categories | 25% |
| 2 | RAGAS evaluation runs successfully | 25% |
| 3 | Baseline metrics captured (faithfulness, relevancy, precision, recall) | 20% |
| 4 | Evaluation report generated as markdown | 15% |
| 5 | Per-question scores identify worst performers | 15% |
