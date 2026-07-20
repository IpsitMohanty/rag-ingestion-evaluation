# LangChain Document Loaders

A working reference for ingesting heterogeneous document formats into LangChain `Document` objects — the loading stage of a RAG pipeline, in isolation from retrieval/generation.

Covers `TextLoader`, `PyPDFLoader`, `PyMuPDFLoader`, `UnstructuredMarkdownLoader`, `JSONLoader`, `CSVLoader`, `UnstructuredCSVLoader`, `WebBaseLoader`, `Docx2txtLoader`, `UnstructuredFileLoader`, and `ArxivLoader` — one notebook, each loader demonstrated against a real file of that type.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
jupyter lab
```

## Windows-specific notes

- `jq` (used by `JSONLoader`'s `jq_schema` parameter) has no prebuilt wheel for Windows and needs a C toolchain to build from source. The notebook replicates the same extraction in plain Python instead — see the "Load from JSON files" section.
- The `arxiv` package's current major version (4.x) restructured its API (`Search.results()` was removed in favor of `Client.results(search)`), which breaks this version of `langchain-community`'s `ArxivLoader`. `requirements.txt` pins `arxiv==2.1.3`, the last release compatible with this loader.

## Origin

Based on a lab notebook from IBM's *Generative AI Applications with RAG and LangChain* course (Coursera). The lab scaffolding and starter code are IBM's; exercises and the Windows compatibility fixes above were completed here.
