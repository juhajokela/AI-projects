import os
import sys

from contextlib import suppress

import streamlit as st

with suppress(ImportError):
    from openai import OpenAI

with suppress(ImportError):
    from hugchat import hugchat
    from hugchat.login import Login


def is_imported(module_name):
    return module_name in sys.modules


def init_openai_chatgpt():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def init_hugchat_bot(email, password):
    cookie_dir_path = '.hugchat'
    cookie_file_path = os.path.join(cookie_dir_path, f'{email}.json')
    sign = Login(email, password)
    if not os.path.exists(cookie_file_path):
        cookies = sign.login(cookie_dir_path=cookie_dir_path, save_cookies=True)
        # return hugchat.ChatBot(cookies=cookies.get_dict())
    return hugchat.ChatBot(cookie_path=cookie_file_path)


is_openai_active = is_imported('openai') and "OPENAI_API_KEY" in st.secrets
is_hugchat_active = is_imported('hugchat') and "HUGGING_FACE_CREDENTIALS" in st.secrets
options = []


if is_openai_active:
    openai_client = init_openai_chatgpt()
    modes = ('ONE-OFF', 'CONVERSATION')
    models = [
        'OpenAI gpt-3.5-turbo',
        'OpenAI gpt-4-turbo-preview',
    ]
    options += [
        model + ' ' + mode
        for model in models
        for mode in modes
    ]


if is_hugchat_active:
    email, password = st.secrets["HUGGING_FACE_CREDENTIALS"].split(':')
    chatbot = init_hugchat_bot(email, password)
    available_models = [
        'google/gemma-7b-it',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        #'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        'meta-llama/Llama-2-70b-chat-hf',
        'codellama/CodeLlama-70b-Instruct-hf',
        #'openchat/openchat-3.5-0106',
    ]
    hugchat_models = [f'HugChat {model}' for model in available_models]
    options += hugchat_models

with st.sidebar:
    st.title("LLM Chat")
    selection = st.selectbox('Mode:', options=options)

provider, model, *mode = selection.split(' ')

# hugchat switch llm model
if provider == 'HugChat':
    idx = hugchat_models.index(selection)
    chatbot.switch_llm(idx)
    chatbot.new_conversation(switch_to=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            f'{message["model"]}\n\n{message["content"]}'
            if 'model' in message else message["content"]
        )

if prompt := st.chat_input("Type..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        if prompt.startswith('/'):
            if prompt == '/list-hugchat-llms':
                st.write(
                    '\n\n'.join(sorted(str(model) for model in chatbot.get_available_llm_models()))
                    if is_hugchat_active else 'HugChat not active!'
                )
        else:

            new_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(new_message)

            if provider == 'OpenAI':
                st.write(f'{model}\n\n')
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ] if mode == ['CONVERSATION'] else [new_message]
                stream = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response, "model": model})
            elif provider == 'HugChat':
                info = chatbot.get_conversation_info()
                st.write(f'{info.model}\n\n')
                stream = chatbot.query(prompt, stream=True)
                response = st.write_stream(chunk['token'] for chunk in stream if chunk is not None)
                st.session_state.messages.append({"role": "assistant", "content": response, "model": info.model})
            else:
                raise NotImplementedError
