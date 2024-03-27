import argparse
import json
import os

from getpass import getpass
from pprint import pformat

import httpx

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from lib import (
    ChatSimple,
    ChatWithStorage,
    ChatWithSummarizedHistory,
)


# arguments
parser = argparse.ArgumentParser(
    prog='LLM Chat',
    description='Chat with LLM',
    epilog='Happy chatting! ;)',
)
parser.add_argument('-log', '--enable-logging', action='store_true')
parser.add_argument('-m', '--mode', choices=['chat', 'chat-storage', 'chat-summarized', 'query'], default='chat')
parser.add_argument('-sid', '--session-id')
parser.add_argument('-llm', '--llm-model', default='gpt-3.5-turbo')
parser.add_argument('-sys', '--system-message')
args = parser.parse_args()
print('args:', args)


def log_request(request):
    print(f"\nRequest: {request.method} {request.url}")
    print(pformat(json.loads(request.content)), '\n')

http_client = httpx.Client(
    event_hooks={
        "request": [log_request],
        #"response": [log_response],
    }
) if args.enable_logging else None
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY') or getpass(),
    model=args.llm_model,
    # https://til.simonwillison.net/httpx/openai-log-requests-responses
    http_client=http_client,
)
output_parser = StrOutputParser()


def run_chat(chat):
    while msg := input('> '):
        chat.query(msg, log=True)


def run_query():
    system_message = [("system", args.system_message)] if args.system_message else []
    prompt = ChatPromptTemplate.from_messages(system_message+[("user", "{question}")])
    chain = prompt | llm | output_parser

    def do_query(chain, question):
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)
        print('')

    while question := input('> '):
        do_query(chain, question)


match args.mode:
    case 'chat':
        chat = ChatSimple(llm, verbose=args.enable_logging)
        run_chat(chat)
    case 'chat-storage':
        chat = ChatWithStorage(
            llm,
            session_id=args.session_id,
            system_message=args.system_message,
        )
        run_chat(chat)
    case 'chat-summarized':
        chat = ChatWithSummarizedHistory(llm, system_message=args.system_message)
        run_chat(chat)
    case 'query':
        run_query()
    case _:
        raise NotImplementedError
