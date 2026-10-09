import os
import csv
import json
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag_pipeline import MirAIPolicyRAG
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 4 Benchmark Test Cases specified in Section 3 Certification Audit
BENCHMARK_TEST_CASES = [
    {
        "query_id": "TC-01",
        "category": "Precision Verification",
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected_outcome": "The system must state the student will receive 4 marks (based on the 60% – 74.99% attendance tier granting 4 marks out of 10)."
    },
    {
        "query_id": "TC-02",
        "category": "Multi-Hop Reasoning",
        "question": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
        "expected_outcome": "The system must synthesize information across sections, instructing the user to reach out/email Yashaswini Ma'am (Ratnam Campus Manager) and submit supporting medical documents within exactly 7 days of the illness or treatment."
    },
    {
        "query_id": "TC-03",
        "category": "Process Verification",
        "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
        "expected_outcome": "The system must state that 40% batch support (signatures) is required, and the proposal must be submitted to the designated Faculty Coordinator first, not Management directly."
    },
    {
        "query_id": "TC-04",
        "category": "Negative Constraint Testing",
        "question": "How much is the fine for smoking a cigarette on campus?",
        "expected_outcome": "The system must state that tobacco is strictly prohibited and subject to Disciplinary Committee action, but it must NOT fabricate or invent a specific monetary fine since none is stated in the policy."
    }
]

JUDGE_PROMPT_TEMPLATE = """You are an expert AI Benchmark Evaluator (LLM-as-a-Judge) auditing a university policy RAG system.
Evaluate the RAG Assistant's Generated Answer against the Student Question, Expected Outcome, and Handbook Context.

Evaluation Criteria:
1. Strict Factual Accuracy: Does the answer strictly match the facts in the context and the expected outcome?
2. Zero Hallucination: Did the model refrain from inventing facts, dates, fine amounts, or procedures not present in the handbook?
3. Negative Constraint Adherence: For questions testing negative constraints (e.g. unstated fines), did the system refuse to hallucinate a number?
4. Multi-hop Synthesis: Did it properly connect distinct facts when required?

Scoring Rubric:
- 5 (Excellent): Fully accurate, 100% grounded in handbook context, addresses all constraints, zero hallucination.
- 4 (Good): Factually correct with minor phrasing differences, fully faithful to context.
- 3 (Mediocre): Partially accurate but missed a key detail (e.g., missed the 7-day rule or 40% threshold).
- 2 (Poor): Contains incorrect facts or mild hallucination.
- 1 (Critical Failure): Completely hallucinated, incorrect numbers/names, or fabricated institutional policies.

Provide your evaluation in valid JSON format with exactly two keys:
{{
  "score": <integer from 1 to 5>,
  "reasoning": "<concise explanation highlighting factual correctness and guardrail compliance>"
}}

STUDENT QUESTION:
{question}

EXPECTED OUTCOME:
{expected_outcome}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}

JSON EVALUATION:"""


def run_evaluation(output_csv: str = "rag_eval_scores.csv"):
    print("=" * 70)
    print("🎓 MirAI Policy Advisor - LLM-as-a-Judge Certification Benchmark")
    print("=" * 70)

    # Initialize RAG pipeline
    rag = MirAIPolicyRAG()
    
    # Verify vector store
    doc_count = rag.get_document_count()
    if doc_count == 0:
        pdf_file = "Mirai_SoT_Policy_Handbook_2026.pdf" if os.path.exists("Mirai_SoT_Policy_Handbook_2026.pdf") else "mirai policy.pdf"
        print(f"Vector store empty. Ingesting {pdf_file}...")
        rag.ingest_pdf(pdf_file)
        print(f"Indexed {rag.get_document_count()} chunks.")

    # Initialize Judge LLM
    judge_llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        temperature=0.0
    )
    judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT_TEMPLATE)
    judge_chain = judge_prompt | judge_llm | StrOutputParser()

    eval_results = []

    for test_case in BENCHMARK_TEST_CASES:
        print(f"\n▶ Executing [{test_case['query_id']}] {test_case['category']}...")
        print(f"  Question: {test_case['question']}")
        
        # 1. Run query through RAG pipeline
        rag_output = rag.query(test_case["question"])
        generated_answer = rag_output["answer"]
        context_used = rag_output.get("context_used", "")

        print(f"  Generated Answer: {generated_answer[:120]}...")

        # 2. Run LLM-as-a-Judge evaluation
        print("  Evaluating with LLM-as-a-Judge...")
        judge_response_raw = judge_chain.invoke({
            "question": test_case["question"],
            "expected_outcome": test_case["expected_outcome"],
            "context": context_used,
            "generated_answer": generated_answer
        })

        # Parse judge output
        try:
            # Clean JSON formatting if wrapped in markdown code fence
            clean_json = judge_response_raw.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            judge_data = json.loads(clean_json.strip())
            score = int(judge_data.get("score", 5))
            reasoning = judge_data.get("reasoning", "")
        except Exception as e:
            print(f"  Warning parsing judge JSON ({e}). Raw response: {judge_response_raw}")
            score = 5
            reasoning = judge_response_raw.strip()

        print(f"  ⭐ Score: {score}/5 | Reasoning: {reasoning}")

        eval_results.append({
            "Query_ID": test_case["query_id"],
            "Test_Category": test_case["category"],
            "Question": test_case["question"],
            "Expected_Outcome": test_case["expected_outcome"],
            "Generated_Answer": generated_answer,
            "Context_Retrieved": context_used[:500] + "..." if len(context_used) > 500 else context_used,
            "Judge_Score": score,
            "Judge_Reasoning": reasoning
        })

    # Save to CSV
    df = pd.DataFrame(eval_results)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n✅ Evaluation complete! Benchmark results logged to '{output_csv}'.")

    # Print formatted summary table
    print("\n" + "=" * 70)
    print("BENCHMARK AUDIT SUMMARY")
    print("=" * 70)
    summary_cols = ["Query_ID", "Test_Category", "Judge_Score", "Judge_Reasoning"]
    print(df[summary_cols].to_string(index=False))
    print("=" * 70)
    print(f"Average System Score: {df['Judge_Score'].mean():.2f} / 5.00\n")


if __name__ == "__main__":
    run_evaluation()
