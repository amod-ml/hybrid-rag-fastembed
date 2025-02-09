# Advanced Retrieval-Augmented Generation (RAG) System with Hybrid Search and Dynamic OCR

This repository contains the final version of a cutting-edge Retrieval-Augmented Generation (RAG) system, developed with a robust, multi-layered architecture tailored to handle diverse university-related queries.

## Project Overview

This RAG system leverages state-of-the-art components to deliver a highly advanced chatbot capable of handling complex document ingestion, hybrid search, and dynamic OCR processing. Notable features include:

- **Dynamic OCR with Table-to-Paragraph Conversion**: Utilizing GPT-4 Vision for Optical Character Recognition, tables are intelligently converted to readable paragraph formats to enhance retrieval accuracy, ensuring all data points are accessible in a non-tabular structure suitable for embedding.
  
- **Hybrid Search with Advanced Embedding Models**:
  - **Dense Embedding**: Incorporates the latest dense embedding model, `Snowflake/Snowflake-Arctic-Embed-XS`, enabling precise vector representation for semantic understanding.
  - **Sparse Embedding**: Utilizes Qdrant's `Qdrant/BM42-All-MiniLM-L6-V2-Attentions`, providing robust sparse embedding for keyword relevance and nuanced information retrieval. Both embeddings are integrated via FastEmbed for a comprehensive search experience.
  
- **Statistical Semantic Chunking**: This system applies OpenAI embeddings for statistical semantic chunking, allowing large documents to be segmented into relevant sections that preserve logical coherence for improved answer retrieval.

- **Comprehensive Hybrid Search Engine**: By combining dense and sparse embeddings, this system enables a sophisticated hybrid search, optimizing retrieval across varied data types and ensuring that results are as relevant and contextually accurate as possible.

- **Additional Features**:
  - Built-in chat history management to maintain conversation continuity.
  - Custom disclaimers based on context source (e.g., proprietary knowledgebase or general AI knowledge).
  - Intelligent query generation tailored for different user types, supporting diverse use cases within a university environment.
