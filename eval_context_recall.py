import os
import pandas as pd
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from ragas import evaluate
from ragas.metrics import context_recall
from ragas.run_config import RunConfig

from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from main import prepare_knowledge_base, get_rag_response


# EVALUATE ONE QUESTION AT A TIME 
run_config = RunConfig(
    max_workers=1,
    timeout=600,  # 10 minutes per question
    max_retries=0
)

llm = LangchainLLMWrapper(
    ChatOllama(
        model="tinyllama",
        temperature=0,
        num_ctx=2048,
        timeout=300  # 5 minutes
    )
)

embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
)

def cap(text, max_chars=200):
    return text[:max_chars] if text else ""

retriever = prepare_knowledge_base(["data/company_policy.pdf"])

QUESTIONS = [
    {
        "question": "What types of leave are available at IIMA and who is eligible for them?",
        "ground_truth": (
            "IIMA provides multiple leave types including Casual Leave, Earned Leave, Half Pay Leave, "
            "Maternity Leave, and Paternity Leave."
        )
    },
    {
        "question": "Describe the promotion policy for Group D employees at IIMA.",
        "ground_truth": (
            "Group D employees may be promoted to Group C through a promotion-with-group-change policy, "
            "based on eligibility, written tests, and interviews as defined in the HR policy."
        )
    }
]

# SAFE RAG RESPONSE WITH TIMEOUT
def safe_get_response(question, retriever):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_rag_response, question, retriever, True)
        try:
            return future.result(timeout=180)
        except TimeoutError:
            return "TIMEOUT_ERROR", []
        except Exception as e:
            return f"ERROR: {str(e)[:50]}", []

# PROCESSING ALL QUESTIONS
all_questions = []
all_answers = []
all_contexts = []
all_ground_truths = []

for q in QUESTIONS:
    print(f"Processing: {q['question'][:50]}...")
    
    answer, sources = safe_get_response(q["question"], retriever)
    
    # CHECK FOR ERRORS BEFORE PROCEEDING
    if "TIMEOUT" in str(answer) or "ERROR" in str(answer):
        print(f"  Warning: {answer}")
        continue
    
    contexts = [cap(doc.page_content) for doc in sources[:4]]
    
    all_questions.append(q["question"])
    all_answers.append(answer)
    all_contexts.append(contexts)
    all_ground_truths.append(q["ground_truth"])
    print(f"  Successfully processed")

if not all_questions:
    print("No valid responses generated. Creating error report...")
    df = pd.DataFrame({
        "question": [q["question"] for q in QUESTIONS],
        "context_recall": [None] * len(QUESTIONS),
        "error": ["All questions failed"] * len(QUESTIONS)
    })
    df.to_csv("context_recall.csv", index=False)
    print("Saved → context_recall.csv (with errors)")
    exit(0)

# EVALUATE QUESTIONS ONE AT A TIME TO PREVENT TIMEOUTS
all_results = []

for i, (question, answer, contexts, ground_truth) in enumerate(zip(all_questions, all_answers, all_contexts, all_ground_truths)):
    print(f"\nEvaluating question {i+1}/{len(all_questions)}...")
    
    # Create dataset for single question
    single_dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    })
    
    try:
        # Evaluate single question with generous timeout
        result = evaluate(
            single_dataset,
            metrics=[context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False
        )
        
        df_single = result.to_pandas()
        all_results.append(df_single)
        
    except Exception as e:
        print(f"  Error evaluating question {i+1}: {e}")
        # Add empty result for this question
        all_results.append(pd.DataFrame({
            "question": [question],
            "context_recall": [None],
            "error": [str(e)[:100]]
        }))

# COMBINE RESULTS
if all_results:
    df = pd.concat(all_results, ignore_index=True)
    
    # Fill missing question column
    if "question" not in df.columns:
        df["question"] = all_questions

    df.to_csv("context_recall_final.csv", index=False)

    print("\nResults Summary:")
    print(f"Questions processed: {len(all_questions)}/{len(QUESTIONS)}")
    if "context_recall" in df.columns:
        valid_scores = df["context_recall"].dropna()
        if len(valid_scores) > 0:
            print(f"Average Context Recall: {valid_scores.mean():.3f}")
    print("Saved → context_recall.csv")