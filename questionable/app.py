import logging
import os
import sys

from contextlib import suppress

from llama_index.core import (
    PromptTemplate,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.llms.openai import OpenAI
import openai
import streamlit as st

import utils


DATA_DIR = './data'
DATA_DIR_IGNORE = [
    '.DS_Store',
]
DB_PATH = './chroma_db'
SYSTEM_PROMPT = """
    You are an expert on the {library} library and your job is to answer technical questions.
    Assume that all questions are related to the {library} library.
    Keep your answers technical and based on facts - do not hallucinate features.
"""

openai.api_key = st.secrets["OPENAI_API_KEY"]
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# 🕵️
st.set_page_config(
    page_title="Questionable",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

with st.sidebar:
    st.title("Questionable 🔍")
    embedding_provider, embedding_model = st.selectbox('Embedding Model:', options=[
        'OpenAI text-embedding-3-small',
        #'OpenAI text-embedding-3-large',
        'OpenAI text-embedding-ada-002',
        'HuggingFace all-MiniLM-L6-v2',
    ]).split(' ')
    inference_provider, inference_model = st.selectbox('Inference Model:', options=[
        'OpenAI gpt-3.5-turbo',
        #'OpenAI gpt-4-turbo-preview',
        #'HuggingFace Writer/camel-5b-hf',
    ]).split(' ')
    options_data = [x for x in os.listdir(DATA_DIR) if x not in DATA_DIR_IGNORE]
    selection_data = st.multiselect('Data:', options=options_data, max_selections=1) # TODO: remove max_selections limit
    print('selection_data:', selection_data)


if embedding_provider == 'OpenAI':
    # @ https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings.html#batch-size
    # By default, embeddings requests are sent to OpenAI in batches of 10.
    # For some users, this may (rarely) incur a rate limit.
    # For other users embedding many documents, this batch size may be too small.
    Settings.embed_model = OpenAIEmbedding(
        model=embedding_model,
        #embed_batch_size=?
    )
if embedding_provider == 'HuggingFace':
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model,
    )
else:
    raise NotImplementedError


#@st.cache_resource(show_spinner=False)
def load_data():
    #import hashlib
    #collection_seed = f'{embedding_provider}_{embedding_model}_{"_".join(sorted(selection_data))}'
    #collection = hashlib.md5(collection_seed.encode('utf-8')).hexdigest()
    #print('collection_seed:', collection_seed)
    #print('collection:', collection)
    collection = f'{embedding_provider}_{embedding_model}_{"_".join(sorted(selection_data))}'
    print('collection:', collection)

    with st.spinner(text="Loading index..."):
        with suppress(ValueError):
            index = utils.load_index_from_db(DB_PATH, collection)
            return index

    with st.spinner(text="Loading and indexing documents, takes about 1-2 minutes..."):
        input_dirs = [os.path.join(DATA_DIR, data_dir) for data_dir in selection_data]
        index = utils.generate_index_as_db(DB_PATH, collection, input_dirs)
        return index


if selection_data:

    if inference_provider == 'OpenAI':
        library = selection_data[0]
        system_prompt = ' '.join(SYSTEM_PROMPT.format(library=library).split())
        Settings.llm = OpenAI(
            model=inference_model,
            temperature=0.5,
            system_prompt=system_prompt,
        )
    elif inference_provider == 'HuggingFace':
        # This will wrap the default prompts that are internal to llama-index
        # taken from https://huggingface.co/Writer/camel-5b-hf
        query_wrapper_prompt = PromptTemplate(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{query_str}\n\n### Response:"
        )
        Settings.llm = HuggingFaceInferenceAPI(
            model_name="Writer/camel-5b-hf",
            tokenizer_name="Writer/camel-5b-hf",
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.25, "do_sample": False},
            query_wrapper_prompt=query_wrapper_prompt,
            device_map="auto",
            tokenizer_kwargs={"max_length": 2048},
        )
        Settings.chunk_size = 512
    else:
        raise NotImplementedError

    index = load_data()
    print('index._embed_model:', index._embed_model)

    if "chat_engine" not in st.session_state.keys():
        # Initialize the chat engine

        # LlamaIndex has four different chat engines:
        #
        # Condense question engine:
        #     Always queries the knowledge base. Can have trouble with meta questions like “What did I previously ask you?”
        # Context chat engine:
        #     Always queries the knowledge base and uses retrieved text from the knowledge base as context for following queries. The retrieved context from previous queries can take up much of the available context for the current query.
        # ReAct agent:
        #     Chooses whether to query the knowledge base or not. Its performance is more dependent on the quality of the LLM. You may need to coerce the chat engine to correctly choose whether to query the knowledge base.
        # OpenAI agent:
        #     Chooses whether to query the knowledge base or not—similar to ReAct agent mode, but uses OpenAI's built-in fuOpenAI'salling capabilities.
        chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True,
            #llm=...,
        )
        print('chat_engine._llm:', chat_engine._llm)
        st.session_state.chat_engine = chat_engine

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            #{"role": "assistant", "content": "Ask me a question about LlamaIndex's open-source Python library!"}
        ]

    if not st.session_state.messages:
        st.info('Ready for questions!')

    # Prompt for user input and save to chat history
    if prompt := st.chat_input("Question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
