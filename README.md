# Verbalens – AI Document Intelligence System

## Overview
Verbalens is a **production-oriented Retrieval-Augmented Generation (RAG) system** that answers business questions from enterprise PDFs with **grounded, explainable responses**.  
The system is designed to reduce **LLM hallucinations** and provide **measurable answer quality**, making it suitable for real-world analytics, compliance, and policy intelligence use cases.

## Business Problem
Organizations rely heavily on documents such as policies, SOPs, and reports, but:
- Analysts misinterpret large documents
- LLMs hallucinate unsupported answers
- Decisions lack traceability to source context
Verbalens addresses this by enforcing **retrieval-grounded answers** and validating output quality using evaluation metrics.

## Key Features
- Retrieval-Augmented Generation (RAG) over single or multiple PDFs  
- Hallucination mitigation through controlled ingestion and retrieval grounding  
- Explainable answers with cited source context  
- Quantitative evaluation using RAGAS metrics  
- Production-ready deployment using FastAPI, Docker, and AWS EC2  

---

## Architecture
PDFs (Default / Uploaded)
↓
Text Cleaning & Chunking
↓
Embeddings + FAISS Vector Store
↓
Retriever (High Recall)
↓
LLM Answer Generation
↓
Evaluated, Grounded Response

## Evaluation
System quality is evaluated using **RAGAS**, focusing on correctness and grounding rather than inflated scores.
Metrics used:
- Answer Relevancy (avg) : 0.96
- Context Precision (avg) : 0.75
- Context Recall (avg) : 0.75
- These metrics ensure answers are supported by retrieved context and reduce hallucinations.

## Tech Stack
- Language: Python  
- LLMs: Groq / Ollama (configurable)  
- Embeddings: HuggingFace Sentence Transformers  
- Vector Store: FAISS  
- Backend: FastAPI  
- Evaluation: RAGAS  
- Deployment: Docker, AWS EC2  

## API Endpoints
| Endpoint | Description |
|--------|-------------|
| GET `/health` | Service health check |
| POST `/upload` | Upload PDFs or index default document |
| POST `/query` | Query indexed documents |

Swagger UI:
http://<EC2_PUBLIC_IP>:8000/docs

## Deployment
- Dockerized FastAPI application  
- Deployed on AWS EC2 (Ubuntu)  
- Uses lazy loading and explicit ingestion to remain stable on low-memory instances
- **Design choice:** Document ingestion and querying are explicitly separated to avoid runtime failures and ensure stability on constrained infrastructure(AWS EC2 instance).
