# RAG Virtual Assistant for the Chemical Industry - Proof of Concept

## Overview
This repository contains the Proof of Concept (PoC) for a virtual assistant based on the Retrieval-Augmented Generation (RAG) architecture, specifically designed for the industrial chemical sector. 

The primary goal is to solve the inefficiency of manual technical information retrieval from unstructured product catalogs (PDFs). The solution utilizes open-source Large Language Models (LLMs) to process natural language queries and provide answers grounded exclusively in internal documents, mitigating the risk of AI hallucinations.

## Motivation
In the modern industrial landscape, agile access to technical information is a critical competitive advantage. In the chemical industry, employees frequently need to consult product specifications, safety standards, and chemical compatibilities. This data is often scattered across extensive and unstructured documents. 

Traditional keyword search is inefficient and prone to human error. By implementing a RAG system, this project combines semantic information retrieval with the generative capabilities of LLMs, enabling users to ask complex questions (e.g., "What are the applications of product X?") and receive accurate, traceable answers. Open-source tools were prioritized to ensure reproducibility and independence from costly proprietary APIs.

## Architecture and Technologies

The solution was implemented in Python, utilizing the Google Colab environment with NVIDIA T4 GPU acceleration.

### Core Stack
*   **Orchestration:** LangChain
*   **Document Loading:** PyMuPDFLoader (chosen for superior text extraction from complex PDFs)
*   **Text Splitting:** RecursiveCharacterTextSplitter (Chunks of 1000 characters with a 150-character overlap)
*   **Embedding Model:** `jinaai/jina-embeddings-v3` (Selected for superior performance in multilingual retrieval and semantic indexing of technical terms in Portuguese)
*   **Generative Model (LLM):** `google/gemma-3-4b-it` (Chosen for excellent instruction following and logical reasoning. Quantized to 4-bits using `bitsandbytes` to fit hardware constraints)

### Prompt Engineering and Memory
*   **System Prompt:** Implemented a Chain-of-Thought strategy. The model is instructed to analyze the question, scan the retrieved context, verify if the information is present, and either generate the answer or admit a lack of knowledge.
*   **Memory:** `ConversationBufferWindowMemory` (k=3) is used to maintain the context of the last 3 interactions, allowing for follow-up questions.

## Dataset and Problem Domain

The domain-specific challenge involves handling chemical terminology and complex PDF structures (tables, double columns). 

For this PoC, a real chemical product catalog from the company Chemyunion (approximately 40 pages) was used. It contains text descriptions, specification tables (INCI Name, applications), and numerical data. However, the pipeline was built to be document-agnostic, allowing the upload and analysis of any technical PDF.

## Results and Limitations

Due to the generative nature of the task and the absence of a labeled Ground Truth dataset, the evaluation was qualitative, focusing on coherence, accuracy, and retrieval capability.

### Successes
*   **Accurate Retrieval:** The pipeline successfully handles direct queries (e.g., "Who is Chemyunion?", "What is Unibase BK?"), retrieving the correct sections and synthesizing coherent responses.
*   **Hallucination Mitigation:** The restrictive prompt strategy proved effective. If the information was not in the text, the model accurately reported that it did not have the information or distinguished between the document's content and its general knowledge.

### Technical Limitations
*   **Hardware Bottlenecks (VRAM):** The Google Colab T4 GPU (16GB VRAM) was insufficient for long conversations. Out of Memory (OOM) errors occurred after an average of 5 interactions due to context accumulation.
*   **Latency:** Inference time was high for production standards, directly resulting from the lack of dedicated, high-tier computational power.
*   **Data Quality (OCR/Extraction):** Complex tables within the PDF were sometimes extracted as disorganized linear text, confirming the "Garbage In, Garbage Out" principle. The LLM's response quality is heavily bounded by the quality of the text extracted by PyMuPDF.

## Future Work

The project successfully demonstrated the technical viability of an open-source RAG assistant for industrial documents. To transition this architecture into a production-ready environment, the following improvements are proposed:

1.  **Infrastructure Scaling:** Upgrade to higher-tier GPUs (e.g., NVIDIA A100) to significantly reduce latency and support much larger context windows.
2.  **Advanced Pre-processing:** Implement dedicated table-OCR and parsing tools (such as Unstructured.io or LlamaParse) to preserve the tabular structure of catalogs during the extraction phase.
3.  **Quantitative Evaluation:** Create a dataset of Question/Answer pairs validated by human domain experts to apply automated evaluation metrics like RAGAS (Retrieval Augmented Generation Assessment).
