import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import context_precision
from ragas.run_config import RunConfig

from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from main import prepare_knowledge_base, get_rag_response


#---------------- CONFIG ----------------
#INCREASED TIMEOUTS
run_config = RunConfig(
    max_workers=1,
    timeout=300,  # Increased from 180
    max_retries=1  # Added retry for stability
)

llm = LangchainLLMWrapper(
    ChatOllama(
        model="llama3:8b",
        temperature=0,
        num_ctx=2048,
        timeout=240  # Increased from 180
    )
)

embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
)

def cap(text, max_chars=200):
    return text[:max_chars] if text else ""


# ---------------- QUESTIONS ----------------
QUESTIONS = [
    {
        "question": "Describe the promotion policy for Group D employees at IIMA.",
        "ground_truth": (
            "Group D employees may be promoted to Group C through a promotion-with-group-change policy, "
            "based on eligibility, written tests, and interviews as defined in the HR policy."
        )
    },
    {
        "question": "What types of leave are available at IIMA and who is eligible for them?",
        "ground_truth": (
            "IIMA provides multiple leave types including Casual Leave, Earned Leave, Half Pay Leave, "
            "Maternity Leave, and Paternity Leave."
        )
    }
]


# ---------------- RAG GENERATION ----------------
retriever = prepare_knowledge_base(["data/company_policy.pdf"])

def safe_get_rag(question):
    """Enhanced with retry logic"""
    for attempt in range(2):  # Try twice
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                get_rag_response,
                question,
                retriever,
                True
            )
            try:
                return future.result(timeout=180 if attempt == 0 else 240)  # Longer on retry
            except TimeoutError:
                if attempt == 0:
                    continue  # Retry once
                return "TIMEOUT_ERROR", []
            except Exception as e:
                return f"ERROR: {str(e)[:50]}", []
    return "MAX_ATTEMPTS_EXCEEDED", []


rows = []

for q in QUESTIONS:
    answer, sources = safe_get_rag(q["question"])
    
    # Checking for various error conditions
    if isinstance(answer, str) and any(err in answer.upper() for err in ["TIMEOUT", "ERROR", "EXCEEDED"]):
        print(f"Skipping question due to: {answer}")
        continue
    
    if not sources:
        continue

    contexts = [cap(doc.page_content) for doc in sources[:3]]

    rows.append({
        "question": q["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": q["ground_truth"]
    })

if not rows:
    print("No valid answers generated for evaluation.")
    # Creating empty result file instead of raising error
    pd.DataFrame(columns=["question", "context_precision"]).to_csv("context_precision_final.csv", index=False)
    print("Saved empty → context_precision_final.csv")
    exit(0)


dataset = Dataset.from_dict({
    "question": [r["question"] for r in rows],
    "answer": [r["answer"] for r in rows],
    "contexts": [r["contexts"] for r in rows],
    "ground_truth": [r["ground_truth"] for r in rows],
})


# ---------------- RAGAS EVALUATION ----------------
try:
    # Wrap evaluation in timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        eval_future = executor.submit(
            evaluate,
            dataset,
            metrics=[context_precision],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False
        )
        result = eval_future.result(timeout=330)  #Slightly longer than run_config
        
    df = result.to_pandas()
    df.to_csv("context_precision_final.csv", index=False)
    
    print("Saved → context_precision_final.csv")
    print("Average Context Precision:", round(df["context_precision"].mean(), 3))
    
except TimeoutError:
    print("RAGAS evaluation timed out. Saving partial results...")
    # Saving what we have
    df = pd.DataFrame({
        "question": [r["question"] for r in rows],
        "context_precision": [None] * len(rows),
        "error": ["Evaluation timeout"] * len(rows)
    })
    df.to_csv("context_precision_final.csv", index=False)
    print("Saved → context_precision_final.csv (with timeout error)")
except Exception as e:
    print(f"Evaluation failed: {e}")
    # Still saving the structure
    df = pd.DataFrame({
        "question": [r["question"] for r in rows],
        "context_precision": [None] * len(rows),
        "error": [str(e)[:100]] * len(rows)
    })
    df.to_csv("context_precision_final.csv", index=False)
    print("Saved → context_precision_final.csv (with error)")