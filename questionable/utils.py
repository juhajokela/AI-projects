import os

import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_files(root_dir):
    result = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            result.append(os.path.join(path, name))
    return result


def collect_input_data(input_dirs=[], input_files=[], recursive=True):
    files = input_files
    for input_dir in input_dirs:
        files += list_files(input_dir) if recursive else os.listdir(input_dir)
    return files


def get_db(db_path):
    chromadb_settings = chromadb.config.Settings(anonymized_telemetry=False)
    db = chromadb.PersistentClient(path=db_path, settings=chromadb_settings)
    return db


def load_index_from_db(db_path, collection):
    db = get_db(db_path)
    # get collection
    chroma_collection = db.get_collection(collection)
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # load index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def generate_index_as_db(db_path, collection, input_dirs=[], input_files=[], recursive=True):
    db = get_db(db_path)
    # https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader.html
    files = collect_input_data(input_dirs, input_files, recursive)
    documents = SimpleDirectoryReader(input_files=files).load_data()
    # create collection
    chroma_collection = db.create_collection(collection)
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # create index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return index


def get_index_from_db(db_path, collection, input_dirs=[], input_files=[], recursive=True):
    try:
        return load_index_from_db(db_path, collection)
    except ValueError: # ValueError: Collection <collection> does not exist.
        return generate_index_as_db(db_path, collection, input_dirs, input_files, recursive)


def load_index_from_file(index_path):
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=index_path)
    )
    return index


def generate_index_as_file(index_path, input_dirs=[], input_files=[], recursive=True):
    files = collect_input_data(input_dirs, input_files, recursive)
    documents = SimpleDirectoryReader(input_files=files).load_data()
    index = VectorStoreIndex.from_documents(documents)
    mkdirs(index_path)
    index.storage_context.persist(persist_dir=index_path)
    return index


def get_index_from_file(index_path, input_dirs=[], input_files=[], recursive=True):
    if os.path.exists(index_path):
        return load_index_from_file(index_path)
    return generate_index_as_file(index_path, input_dirs, input_files, recursive)

'''
db_path = './chroma_db'
collection = 'streamlit'
data_dir = "./data/streamlit"
index = load_index_from_db(db_path, collection)
index = generate_index_as_db(db_path, collection, input_dirs=[data_dir])
index = get_index_from_db(db_path, collection, input_dirs=[data_dir])

index_path = './indexes/streamlit'
data_dir = "./data/streamlit"
index = load_index_from_file(index_path)
index = generate_index_as_file(index_path, input_dirs=[data_dir])
index = get_index_from_file(index_path, input_dirs=["./data/streamlit"])
'''