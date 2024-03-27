# querify

Different RAG implementations based on LangChain, using FAISS as a vector database.
See [docs/rag-log.txt](docs/rag-log.txt) for results.

### Build

```
./docker-scripts/docker-build.sh
```

### Run

```
./docker-scripts/docker-shell.sh
```

##### Chat

```
python apps/chat/app.py -log
python apps/chat/app.py -log -m chat
python apps/chat/app.py -log -m chat-storage
python apps/chat/app.py -log -m chat-storage -sid random
python apps/chat/app.py -log -m chat-summarized
python apps/chat/app.py -log -m query
```

##### RAG

```
python apps/rag/app.py langsmith url=https://docs.smith.langchain.com/user_guide
> How can LangSmith help with testing?
> Tell me how
```

```
python apps/rag/app.py -log -emb=OpenAI typechat txt=data/typechat
> Tell me about TypeChat
```

```
python apps/rag/app.py -log -emb=OpenAI streamlit md=data/streamlit
> Tell me about Streamlit
```

```
python apps/rag/app.py -log -emb=OpenAI -m query-advanced langsmith-openai url=https://docs.smith.langchain.com/user_guide
> How can LangSmith help with testing?
> Tell me how
```

```
python apps/rag/app.py -log -emb=OpenAI -m chat-history langsmith-openai url=https://docs.smith.langchain.com/user_guide
> How can LangSmith help with testing?
> Tell me how
```

```
python apps/rag/app.py -log -emb=OpenAI -m chat-memory langsmith-openai url=https://docs.smith.langchain.com/user_guide
> How can LangSmith help with testing?
> Tell me how
```

### TODO

- apps/rag.py loaders:
    - from langchain_community.document_loaders import UnstructuredPDFLoader
    - document_loaders.csv_loader.UnstructuredCSVLoader
    - document_loaders.excel.UnstructuredExcelLoader
    - document_loaders.powerpoint.UnstructuredPowerPointLoader
    - document_loaders.word_document.UnstructuredWordDocumentLoader
    - JSON
    - CSV ("structured")
    - for more, see: https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.document_loaders
- https://api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html#langchain_experimental.text_splitter.SemanticChunker
- SemanticChunker(breakpoint_threshold_type='percentile'/'standard_deviation'/'interquartile')
- FAISS (https://python.langchain.com/docs/integrations/vectorstores/faiss)
    - vector_db.similarity_search_with_score("...")
    - vector_db.merge_from(another_db)
    - vector_db.delete([vector_db.index_to_docstore_id[idx]])
### Tips

```
from langchain_community.document_loaders import JSONLoader

# Common JSON structures with jq schema:
#
# JSON        -> [{"text": ...}, {"text": ...}, {"text": ...}]
# jq_schema   -> ".[].text"
#
# JSON        -> {"key": [{"text": ...}, {"text": ...}, {"text": ...}]}
# jq_schema   -> ".key[].text"
#
# JSON        -> ["...", "...", "..."]
# jq_schema   -> ".[]"

# Define a metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["key"] = record.get("key")
    return metadata

JSONLoader(
    ...,
    jq_schema=...,
    metadata_func=metadata_func,
)
```
