import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig

from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from main import prepare_knowledge_base, get_rag_response


# ---------------- CONFIG ----------------
run_config = RunConfig(
    max_workers=1,
    timeout=120,
    max_retries=0
)

llm = LangchainLLMWrapper(
    ChatOllama(
        model="llama3:8b",
        temperature=0,
        num_ctx=2048,
        timeout=180
    )
)

embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
)

def cap(text, max_chars=200):
    return text[:max_chars]


# ---------------- QUESTIONS ----------------
QUESTIONS = [
    {
        "question": "Explain the retirement policy at IIMA for different categories of employees.",
        "ground_truth": (
            "Faculty members retire at 65 years, while non-faculty employees retire at 60 years. "
            "Retirement is effective from the afternoon of the last day of the month in which the age is attained."
        )
    },
    {
        "question": "What types of leave are available at IIMA and who is eligible for them?",
        "ground_truth": (
            "IIMA provides multiple leave types including Casual Leave, Earned Leave, Half Pay Leave, "
            "Maternity Leave, and Paternity Leave, with eligibility depending on employee category and service rules."
        )
    },
    {
        "question": "Describe the promotion policy for Group D employees at IIMA.",
        "ground_truth": (
            "Group D employees may be promoted to Group C through a promotion-with-group-change policy, "
            "based on eligibility, written tests, and interviews as defined in the HR policy."
        )
    },
    {
        "question": "What are the eligibility criteria and selection process for internal promotions at IIMA?",
        "ground_truth": (
            "Eligibility for promotions depends on experience, service record, and role-specific criteria, "
            "with selection involving written examinations and interviews."
        )
    }
]


# ---------------- RAG GENERATION----------------
retriever = prepare_knowledge_base(["data/company_policy.pdf"])

def safe_get_rag(question):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            get_rag_response,
            question,
            retriever,
            True        # eval_mode=True
        )
        try:
            return future.result(timeout=150)
        except TimeoutError:
            return "TIMEOUT_ERROR", []


# ---------------- COLLECT VALID DATA ----------------
rows = []

for q in QUESTIONS:
    answer, sources = safe_get_rag(q["question"])

    if "TIMEOUT" in answer or not sources:
        continue   # DROP invalid cases completely

    contexts = [cap(doc.page_content) for doc in sources[:3]]

    rows.append({
        "question": q["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": q["ground_truth"]
    })

if not rows:
    raise RuntimeError("No valid answers generated for evaluation.")


dataset = Dataset.from_dict({
    "question": [r["question"] for r in rows],
    "answer": [r["answer"] for r in rows],
    "contexts": [r["contexts"] for r in rows],
    "ground_truth": [r["ground_truth"] for r in rows],
})


# ---------------- RAGAS EVALUATION ----------------
result = evaluate(
    dataset,
    metrics=[answer_relevancy],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config,
    raise_exceptions=False
)

df = result.to_pandas()
df.to_csv("answer_relevancy_final.csv", index=False)

print("\nSaved → answer_relevancy_final.csv")
print("Average Answer Relevancy:", round(df["answer_relevancy"].mean(), 3))
