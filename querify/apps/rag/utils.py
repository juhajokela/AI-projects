import os
from typing import (
    Sequence,
)

from langchain_community.document_loaders import (
    BSHTMLLoader,
    OnlinePDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from definitions import Texts


def clean_prompt(original_prompt):
    lines = original_prompt.splitlines()
    cleaned_lines = (line.strip() for line in lines)
    joined_lines = '\n'.join(cleaned_lines)
    cleaned_prompt = joined_lines.strip()
    return cleaned_prompt


def pack_documents(texts: Sequence[str], metadata: Sequence[dict] = []) -> Sequence[Document]:
    return [Document(page_content=t, metadata=m) for t, m in zip(texts, metadata)]


def documents_to_texts(documents: Sequence[Document]) -> Texts:
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    return Texts(texts, metadatas)


def load(source, loader_cls=UnstructuredFileLoader, splitter=None, skip_errors=False, **kwargs):
    kwargs.setdefault('show_progress', True)
    loader = DirectoryLoader(
        source,
        loader_cls=loader_cls,
        silent_errors=skip_errors,
        **kwargs
    ) if os.path.isdir(source) else loader_cls(source)
    if splitter:
        return loader.load_and_split(splitter)
    return loader.load()


def load_data(source, format=None):

    if format == 'url':
        if source.endswith('.pdf'):
            return OnlinePDFLoader(source).load_and_split()
        return WebBaseLoader(source).load_and_split()

    if format == 'txt':
        splitter = RecursiveCharacterTextSplitter()
        return load(
            source,
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True},
            splitter=splitter,
            glob='**/*.txt',
        )

    if format == 'html':
        return load(source, loader_cls=BSHTMLLoader, glob='**/*.html')

    if format == 'md':
        return load(
            source,
            loader_cls=UnstructuredMarkdownLoader,
            loader_kwargs={'mode': 'elements'},
            glob='**/*.md',
        )

    if format == 'md':
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
        )
        return load(source, splitter=splitter, glob='**/*.md')

    splitter = RecursiveCharacterTextSplitter()
    return load(source, splitter=splitter)


def load_db(db_path, embedding_function, data_sources):
    if os.path.exists(db_path):
        print('LOADING VECTOR DB!')
        vector_db = FAISS.load_local(db_path, embedding_function, allow_dangerous_deserialization=True)
    else:
        print('GENERATING VECTOR DB!')
        documents = []
        for data_source in data_sources:
            documents.extend(load_data(*data_source))
        vector_db = FAISS.from_documents(documents, embedding_function)
        vector_db.save_local(db_path)

    print('vector_db.index.ntotal:', vector_db.index.ntotal)
    return vector_db
