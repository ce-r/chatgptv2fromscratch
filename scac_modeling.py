# -*- coding: utf-8 -*-
"""scac_modeling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nofhpQUcMV_EK9s06DQZTOnf-OJN2uaj
"""

# !huggingface-cli login
# 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

!pip install pypdf
!pip install SQLAlchemy==1.4.49

!pip install fitz
!pip install tools
!pip install bs4

!pip install faiss-cpu
!pip install farm-haystack[sentence-transformers]

!pip install pdfplumber

!pip install transformers datasets
!pip install -U scikit-learn

!pip uninstall -y pydantic
!pip install pydantic==1.10.13

from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
import json
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.schema import Document
from typing import List, Optional
import logging
import warnings
import pickle

import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import faiss
import pickle
from tqdm import tqdm
from copy import deepcopy
# from accelerate import Accelerator

import time
import pdfplumber
import gc
from transformers import T5Tokenizer, T5ForConditionalGeneration

logging.getLogger("pdfminer").setLevel(logging.ERROR)

DOCKETS_DIR = Path("/content/drive/MyDrive/dockets_test")
CSV_PATH = "scac-filings4db.csv"
FAISS_INDEX_PATH = "full_faiss_index"
LOG_FILE = "generation_log.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".html"]

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto")

# Utils
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def case_exists(df, case_id):
    return case_id in df["case_id"].valueso

def save_prompt_and_output(prompt, response, task, case_id, file_name, idx):
    with open(f"input_{task}_{case_id}_{file_name}_chunk{idx}.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(f"output_{task}_{case_id}_{file_name}_chunk{idx}.txt", "w", encoding="utf-8") as f:
        f.write(response)

# Text Extraction
def extract_text(file_path: Path):
    if file_path.suffix == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.suffix == ".html":
        return BeautifulSoup(file_path.read_text(encoding="utf-8", errors="ignore"), "html.parser").get_text()
    elif file_path.suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    return ""

def preprocess_text(text):
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove headers/footers if possible
    lines = [line for line in text.split('\n') if len(line.strip()) > 30]
    return '\n'.join(lines)

# Chunking
def chunk_text(text, max_tokens=1024, stride=128):
    """Proper text chunking with overlap and boundary handling"""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, total_tokens, max_tokens - stride):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

# Prompts
def build_prompt(task, text):
    examples = {
        "violations": (
            "From the legal document below, explicitly identify all cited SEC violation references, "
            "specifying exact sections and rules (e.g., 'Section 10(b) of the Securities Exchange Act of 1934, Rule 10b-5'). "
            "If no violations are cited, respond strictly with 'None'.\n\n"
            f"Document:\n{text}\n\n"
            "Violations:"
        ),
        "settlement_amount": (
            "From the legal document below, extract explicitly stated settlement amounts directly associated with "
            "the terms 'settlement', 'settled', or 'judgment'. "
            "Respond ONLY with exact monetary amounts (e.g., '$250 million', '$1 million'). If none are present, respond 'None'.\n\n"
            f"Document:\n{text}\n\n"
            "Amount:"
        )
    }
    return examples[task]

# function to safely chunk prompt+text
def build_chunked_prompts(task, text, max_tokens=512, overlap=50):
    base_prompt = build_prompt(task, "")
    base_tokens = tokenizer.encode(base_prompt, return_tensors="pt")[0]
    base_length = len(base_tokens)

    text_tokens = tokenizer.encode(text, return_tensors="pt")[0]
    text_max_length = max_tokens - base_length - 10  # buffer for safety

    chunks = []
    start = 0
    while start < len(text_tokens):
        end = min(start + text_max_length, len(text_tokens))
        chunk = text_tokens[start:end]
        combined = torch.cat([base_tokens, chunk])
        chunks.append(tokenizer.decode(combined, skip_special_tokens=True))
        start += text_max_length - overlap
    return chunks

# Generation with Cleanup, Timeout, Logging
def safe_generate(prompt, tokenizer, model, max_input_tokens=512, timeout_sec=60):
    def _generate():
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    do_sample=False,  # deterministic, structured output
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                return tokenizer.decode(output[0], skip_special_tokens=True).strip()
        except Exception as e:
            return f"[ERROR] {str(e)}"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_generate)
        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            print("[TIMEOUT] Generation took too long.")
            return "[TIMEOUT]"

def is_quality_response(response):
    response = response.lower().strip()
    if response == "none" or response == "[error]" or response == "[timeout]" or not response:
        return False
    if response.startswith("case no.") or "court" in response:
        return False
    return True

# Transformer-Only Pipeline
def transformer_pipeline(df, tokenizer, model, DOCKETS_DIR, output_csv="structured_output_transformer.csv"):
    """Process legal documents to extract violations and settlement amounts using transformer model."""

    # Initialize counters and logging
    processed_cases = 0
    successful_extractions = 0
    start_time = time.time()

    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    for ticker_dir in tqdm(list(DOCKETS_DIR.iterdir()), desc="Processing Ticker Folders"):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        for filing_dir in ticker_dir.iterdir():
            if not filing_dir.is_dir():
                continue
            filing_date = filing_dir.name
            case_id = f"{ticker}_{filing_date}"

            if case_id not in df["case_id"].values:
                continue

            processed_cases += 1
            print(f"\n[PROCESSING CASE] {case_id}")

            best_outputs = {"violations": None, "settlement_amount": None}
            field_hits = {"violations": False, "settlement_amount": False}

            for doc_dir in filing_dir.iterdir():
                if not doc_dir.is_dir():
                    continue
                files = list(doc_dir.glob("*"))
                if not files:
                    continue

                file = files[0]  # Process first file in directory
                try:
                    text = extract_text(file)
                    text = preprocess_text(text)
                    if not text.strip():
                        continue

                    print(f"[PROCESSING] {case_id} -> Document: {file.name}")

                    for task in ["violations", "settlement_amount"]:
                        if field_hits[task]:
                            continue

                        chunks = build_chunked_prompts(task, text)
                        for idx, prompt in enumerate(chunks):
                            # Validate input length
                            token_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
                            print(f"[DEBUG] Chunk {idx} -> token count: {token_len}")

                            if token_len > 512: # 2048
                                print(f"[SKIP] Chunk {idx} too long for model context window.")
                                continue

                            # Save prompt for debugging
                            with open(f"input_{task}_{case_id}_{file.name}_chunk{idx}.txt", "w", encoding="utf-8") as f:
                                f.write(prompt)

                            try:
                                # Generate with safer parameters
                                response = safe_generate(prompt, tokenizer, model)

                                # Save response
                                with open(f"output_{task}_{case_id}_{file.name}_chunk{idx}.txt", "w", encoding="utf-8") as f:
                                    f.write(response)

                            except Exception as e:
                                # print(f"[ERROR] Generation failed: {str(e)}")
                                response = "None"
                            finally:
                                torch.cuda.empty_cache()
                                gc.collect()

                            # Save response
                            with open(f"output_{task}_{case_id}_{file.name}_chunk{idx}.txt", "w", encoding="utf-8") as f:
                                f.write(response)

                            save_prompt_and_output(prompt, response, task, case_id, file.name, idx)

                            if is_quality_response(response):
                                best_outputs[task] = response
                                field_hits[task] = True
                                print(f"[SUCCESS] Found {task} for {case_id}: {response}")
                                break

                except Exception as e:
                    print(f"[ERROR] Processing failed for {file.name}: {str(e)}")
                    continue

            # Update DataFrame if we found any results
            if any(best_outputs.values()):
                idx = df[df["case_id"] == case_id].index[0]
                for task in ["violations", "settlement_amount"]:
                    if best_outputs[task]:
                        df.at[idx, task] = best_outputs[task]
                        successful_extractions += 1

                # Save progress periodically
                df.to_csv(output_csv, index=False)
                print(f"[PROGRESS] Saved results for {case_id}")

    # Final save and statistics
    total_time = time.time() - start_time
    df.to_csv(output_csv, index=False)

    print("\n[SUMMARY]")
    print(f"  Processed cases: {processed_cases}")
    print(f"  Successful extractions: {successful_extractions}")
    print(f"  Total runtime: {total_time:.2f} seconds")
    print(f"[DONE] Results saved to {output_csv}")

    return df

"""# ----------------------------------------------------------
# RAG

# We only need to worry about how the document store is created insofar as it aligns with how the retriever embeds the query.
"""

def load_document(fpath: str):
    ext = os.path.splitext(fpath)[1].lower()
    try:
        if ext == ".txt":
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            reader = PdfReader(fpath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".html":
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
                return soup.get_text(separator="\n")
    except Exception as e:
        print(f"[WARN] Failed to load {fpath}: {e}")
        return None

# Root directory of docket folders
DOC_ROOT = "/content/drive/MyDrive/dockets_test"
# Output path for FAISS index and docs
FAISS_INDEX_PATH = "/content/full_faiss_index"
# Transformer model to use for embeddings (default legal-compatible)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "nlpaueb/legal-bert-small-uncased"  # Alternate for legal context
# File types to accept during loading
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".html"]

def build_document_store():
    """Build FAISS document store and index, grouped by case_id."""
    doc_store = FAISSDocumentStore(
        embedding_dim=384,
        faiss_index_factory_str="Flat",
        sql_url="sqlite:///rag_legal.db"
    )

    retriever = EmbeddingRetriever(
        document_store=doc_store,
        embedding_model=EMBEDDING_MODEL,
        model_format="sentence_transformers",
        use_gpu=torch.cuda.is_available(),
        progress_bar=True
    )

    documents = []
    for root, _, files in os.walk(DOC_ROOT):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            path_parts = Path(fpath).relative_to(DOC_ROOT).parts
            if len(path_parts) >= 3:
                ticker, filing_date, doc_title = path_parts[:3]
                case_id = f"{ticker}_{filing_date}"
            else:
                continue

            text = load_document(fpath)
            if not text or not text.strip():
                continue

            documents.append(Document(
                content=text,
                meta={
                    "case_id": case_id,
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "doc_title": doc_title,
                    "source": fname,
                    "path": fpath
                }
            ))

    print(f"[INFO] Loaded {len(documents)} documents")

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_length=150,
        split_overlap=30,
        split_respect_sentence_boundary=True,
        language="en"
    )

    chunks = preprocessor.process(documents)
    print(f"[INFO] Preprocessed into {len(chunks)} chunks")

    doc_store.write_documents(chunks)
    doc_store.update_embeddings(retriever)
    print(f"[INFO] FAISS embedding count: {doc_store.get_embedding_count()}")

    docs_all = doc_store.get_all_documents()

    # Build index map by case_id
    index_map = {}
    for idx, doc in enumerate(docs_all):
        case_id = doc.meta.get("case_id")
        if case_id:
            index_map.setdefault(case_id, []).append(idx)

    # Save everything
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    with open(os.path.join(FAISS_INDEX_PATH, "docs.pkl"), "wb") as f:
        pickle.dump(docs_all, f)

    with open(os.path.join(FAISS_INDEX_PATH, "faiss_index.json"), "w") as f:
        json.dump(index_map, f)

    doc_store.save(
        index_path=os.path.join(FAISS_INDEX_PATH, "faiss_index.faiss"),
        config_path=os.path.join(FAISS_INDEX_PATH, "faiss_config.json")
    )

    print(f"[DONE] Saved index, docs, and case_id map to '{FAISS_INDEX_PATH}'")
    return doc_store, docs_all, index_map, retriever

# Load/Build RAG Components
def create_rag_components(
    index_dir="full_faiss_index",
    doc_root="/content/drive/MyDrive/dockets_test",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"):

    # Paths
    docs_path = os.path.join(index_dir, "docs.pkl")
    faiss_index_path = os.path.join(index_dir, "faiss_index.faiss")
    faiss_config_path = os.path.join(index_dir, "faiss_config.json")
    index_map_path = os.path.join(index_dir, "faiss_index.json")

    # -----------------------------------------------
    # If all files exist, load from disk
    # -----------------------------------------------
    if all(os.path.exists(p) for p in [docs_path, faiss_index_path, faiss_config_path, index_map_path]):
        print("[INFO] Loading existing FAISS index and retriever...")

        # Load documents
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)

        # Load index map
        with open(index_map_path, "r") as f:
            index_map = json.load(f)

        # Load FAISS index from disk
        document_store = FAISSDocumentStore.load(
            index_path=faiss_index_path,
            config_path=faiss_config_path
        )

        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=embedding_model,
            use_gpu=torch.cuda.is_available()
        )

        index = faiss.read_index(faiss_index_path)

        print(f"[INFO] Loaded {len(documents)} documents and {len(index_map)} FAISS index entries.")
        return documents, index, index_map, retriever

    # -----------------------------------------------
    # Otherwise, build from scratch
    # -----------------------------------------------
    else:
        print("[INFO] FAISS components not found. Building from scratch...")

        os.makedirs(index_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Supported extensions
        SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".html"]

        # Initialize doc store + retriever
        doc_store = FAISSDocumentStore(
            embedding_dim=384,
            faiss_index_factory_str="Flat",
            sql_url="sqlite:///rag_legal.db"
        )

        retriever = EmbeddingRetriever(
            document_store=doc_store,
            embedding_model=embedding_model,
            model_format="sentence_transformers",
            use_gpu=torch.cuda.is_available(),
            progress_bar=True
        )

        # Load and structure documents
        documents = []
        for root, _, files in os.walk(doc_root):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                fpath = os.path.join(root, fname)
                path_parts = Path(fpath).relative_to(doc_root).parts
                if len(path_parts) >= 3:
                    ticker, filing_date, doc_title = path_parts[:3]
                    case_id = f"{ticker}_{filing_date}"
                else:
                    logger.warning(f"Unexpected path structure: {fpath}")
                    continue

                text = load_document(fpath)
                if not text or not text.strip():
                    continue

                documents.append(Document(
                    content=text,
                    meta={
                        "case_id": case_id,
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "doc_title": doc_title,
                        "source": fname,
                        "path": fpath
                    }
                ))

        logger.info(f"Loaded {len(documents)} documents.")

        # Preprocessing
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_length=150,
            split_overlap=30,
            split_respect_sentence_boundary=True,
            language="en"
        )
        chunks = preprocessor.process(documents)
        logger.info(f"Created {len(chunks)} chunks.")

        # Indexing
        doc_store.write_documents(chunks)
        doc_store.update_embeddings(retriever)
        print(f"Documents in DB: {doc_store.get_document_count()}")
        print(f"Embeddings in FAISS: {doc_store.get_embedding_count()}")

        embedding_count = doc_store.get_embedding_count()
        doc_count = doc_store.get_document_count()
        logger.info(f"Embedding count: {embedding_count} | Document count: {doc_count}")

        if embedding_count == 0:
            raise RuntimeError("Embedding update failed. No embeddings were written to FAISS.")
        if embedding_count != doc_count:
            raise RuntimeError("Mismatch: Number of embeddings does not match number of documents. Aborting save.")

        # Save to disk
        docs_all = doc_store.get_all_documents()
        with open(docs_path, "wb") as f:
            pickle.dump(docs_all, f)

        index_map = {}
        for idx, doc in enumerate(docs_all):
            cid = doc.meta.get("case_id")
            if cid:
                index_map.setdefault(cid, []).append(idx)

        with open(index_map_path, "w") as f:
            json.dump(index_map, f)

        doc_store.save(
            index_path=faiss_index_path,
            config_path=faiss_config_path
        )

        index = faiss.read_index(faiss_index_path)

        print(f"[DONE] Saved FAISS index, docs, and map to '{index_dir}'")
        # return docs_all, index, index_map, retriever
        return documents, index, index_map, retriever.model if hasattr(retriever, "model") else retriever

"""# RAG + Transformer"""

def encode_query(query, encoder):
    return encoder.encode(query, convert_to_tensor=False, normalize_embeddings=True)

def get_doc_texts_from_indices(indices, documents):
    texts = []
    for idx in indices:
        if 0 <= idx < len(documents):
            doc = documents[idx]
            text = getattr(doc, "content", "") if hasattr(doc, "content") else doc.get("content", "")
            if text:
                texts.append(text)
        # else:
        #     print(f"[WARN] Invalid FAISS index: {idx}")
    return texts

"""# EmbeddingRetriever does not have an .encode() method, only SentenceTransformer models do."""

def rag_transformer_inference(
    df,
    tokenizer,
    model,
    encoder,
    documents,
    faiss_index,
    index_map,
    top_k=3,
    max_length=512,
    output_csv="structured_output_rag.csv"):

    num_violations_filled = 0
    num_settlements_filled = 0
    start_time = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="RAG Inference"):
        case_id = row["case_id"]
        if pd.notna(row["violations"]) and pd.notna(row["settlement_amount"]):
            continue

        if case_id not in index_map:
            continue

        retrieved_chunks = get_doc_texts_from_indices(index_map[case_id], documents)
        if not retrieved_chunks:
            continue

        # Join all chunks to simulate a full document
        full_text = " ".join(retrieved_chunks)
        full_text = preprocess_text(full_text)

        best_outputs = {"violations": None, "settlement_amount": None}
        field_hits = {"violations": False, "settlement_amount": False}

        for task in ["violations", "settlement_amount"]:
            if field_hits[task]:
                continue

            prompt_chunks = build_chunked_prompts(task, full_text)

            for idx, prompt in enumerate(prompt_chunks):
                token_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
                if token_len > 512:
                    continue

                response = safe_generate(prompt, tokenizer, model, max_input_tokens=max_length)
                save_prompt_and_output(prompt, response, task, case_id, f"RAG_virtual.txt", idx)

                if is_quality_response(response):
                    best_outputs[task] = response
                    field_hits[task] = True
                    print(f"[SUCCESS][RAG] {task} for {case_id}: {response}")
                    break

        if any(best_outputs.values()):
            idx = df[df["case_id"] == case_id].index[0]
            for task in ["violations", "settlement_amount"]:
                if best_outputs[task]:
                    df.at[idx, task] = best_outputs[task]
                    if task == "violations":
                        num_violations_filled += 1
                    elif task == "settlement_amount":
                        num_settlements_filled += 1

    total_time = round(time.time() - start_time, 2)
    df.to_csv(output_csv, index=False)

    print("\n[RAG + Transformer]")
    print(f"  Violations filled: {num_violations_filled}/{len(df)}")
    print(f"  Settlements filled: {num_settlements_filled}/{len(df)}")
    print(f"  Runtime: {total_time:.2f} seconds")
    print(f"[DONE] RAG output saved -> {output_csv}")

"""# Comparison"""

# Summary helper
def summarize(df, label, elapsed_time):
    v_filled = df["violations"].notna().sum()
    s_filled = df["settlement_amount"].notna().sum()
    total = len(df)
    print(f"\n[{label}]")
    print(f"  Violations filled: {v_filled}/{total}")
    print(f"  Settlements filled: {s_filled}/{total}")
    print(f"  Runtime: {elapsed_time:.2f} seconds")
    return {
        "method": label,
        "violations_filled": v_filled,
        "settlement_filled": s_filled,
        "total": total,
        "runtime_sec": round(elapsed_time, 2)
    }

# Compare Transformer w/ RAG Pipelines
def compare_extraction_pipelines(
    df_original,
    DOCKETS_DIR,
    tokenizer,
    model,
    retriever,
    encoder,
    rag_documents,
    faiss_index,
    index_map,
    output_transformer="structured_output_transformer.csv",
    output_rag="structured_output_rag.csv"):

    for col in ["violations", "settlement_amount"]:
      if col not in df_original.columns:
          df_original[col] = None

    # Make copies to avoid modifying the original DataFrame
    df_transformer = deepcopy(df_original)
    df_rag = deepcopy(df_original)

    # Run Transformer-Only pipeline
    print("\n[T-INFO] Running Transformer-Only Pipeline...")
    start_t = time.time()
    transformer_pipeline(df_transformer, tokenizer, model, DOCKETS_DIR)
    time_transformer = time.time() - start_t

    # Run RAG + Transformer pipeline
    print("\n[RT-INFO] Running RAG + Transformer Pipeline...")
    start_r = time.time()
    # encoder = SentenceTransformer("all-MiniLM-L6-v2") # VERY BAD

    # Must use same encoder used for FAISS document embeddings
    rag_transformer_inference(
        df=df_rag,
        tokenizer=tokenizer,
        model=model,
        encoder=encoder,
        documents=rag_documents,
        faiss_index=faiss_index,
        index_map=index_map,
        output_csv="structured_output_rag.csv"
    )
    time_rag = time.time() - start_r

    # Collect results
    summary = [summarize(df_transformer, "Transformer-Only", time_transformer),
               summarize(df_rag, "RAG + Transformer", time_rag)]

    return pd.DataFrame(summary)

# !rm -rf full_faiss_index
# !rm -f rag_legal.db

documents, index, index_map, retriever = create_rag_components()
# encoder = retriever.model if hasattr(retriever, "model") else retriever
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
df = pd.read_csv(CSV_PATH) # scac-filings4db.csv

comparison_df = compare_extraction_pipelines(
    df_original=df,
    DOCKETS_DIR=DOCKETS_DIR,
    tokenizer=tokenizer,
    model=model,
    retriever=retriever,
    encoder=encoder,
    rag_documents=documents,
    faiss_index=index,
    index_map=index_map
)
comparison_df.to_csv("pipeline_comparison_results.csv", index=False)

