import argparse
import logging
import os
import sys

from getpass import getpass

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from definitions import DataSource
from openai_utils import openai_http_client
from utils import load_db


# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

EMBEDDING_FUNCTIONS = {
    'all-MiniLM-L6-v2': lambda: SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    'OpenAI': lambda: OpenAIEmbeddings(),
}
def get_embedding_function(embedding_function):
    embedding_function_loader = EMBEDDING_FUNCTIONS[embedding_function]
    return embedding_function_loader()


def parse_data_source(data_source: str) -> DataSource:
    source, *format = reversed(data_source.split('='))
    assert 0 < len(source)
    if format:
        assert format[0] in ['url', 'txt', 'md', 'html'], format[0]
    return DataSource(source, format[0] if format else None)


parser = argparse.ArgumentParser(
    prog='RAG',
    description='A runner for different RAG implementations',
    epilog="Let's go! ;)",
)
parser.add_argument('database')
parser.add_argument('data_sources', nargs='+')
parser.add_argument('-m', '--mode', choices=['chat', 'chat-history', 'chat-memory', 'query', 'query-advanced'], default='chat')
parser.add_argument('-log', '--enable-logging', action='store_true')
parser.add_argument('-emb', '--embedding', choices=list(EMBEDDING_FUNCTIONS.keys()), default='all-MiniLM-L6-v2')
parser.add_argument('-llm', '--llm-model', default='gpt-3.5-turbo')
args = parser.parse_args()
print('args:', args)

if args.enable_logging:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

db_path = os.path.join('vector_dbs', args.database)
embedding_function = get_embedding_function(args.embedding)
data_sources = [parse_data_source(src) for src in args.data_sources]
vector_db = load_db(db_path, embedding_function, data_sources)
retriever = vector_db.as_retriever()
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY') or getpass(),
    model=args.llm_model,
    # https://til.simonwillison.net/httpx/openai-log-requests-responses
    http_client=openai_http_client if args.enable_logging else None,
)


def run_chat(chat_cls):
    chat = chat_cls(llm, retriever)
    while msg := input('> '):
        chat.query(msg, log=True)


def run_query(chain, query):
    while msg := input('> '):
        query(chain, msg, log=True)


from lib import (
    Chat,
    ChatWithHistory,
    ChatWithMemory,
    create_advanced_chain,
    create_simple_chain,
    query_advanced_chain,
    query_simple_chain,
)


match args.mode:
    case 'chat':
        run_chat(Chat)
    case 'chat-history':
        run_chat(ChatWithHistory)
    case 'chat-memory':
        run_chat(ChatWithMemory)
    case 'query':
        chain = create_simple_chain(llm, retriever)
        run_query(chain, query_simple_chain)
    case 'query-advanced':
        chain = create_advanced_chain(llm, retriever)
        run_query(chain, query_advanced_chain)
    case _:
        raise NotImplementedError
