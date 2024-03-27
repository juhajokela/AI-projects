from abc import (
    ABC,
    abstractmethod,
)

from langchain.chains import ConversationChain
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
)
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


class BaseChat(ABC):

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @abstractmethod
    def query(self, question: str):
        pass


class ChatSimple(BaseChat):

    def __init__(self, *args, verbose=False):
        super().__init__(*args)
        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=verbose,
            memory=ConversationBufferMemory()
        )

    def query(self, question, log=False):
        reply = self.conversation.predict(input=question)
        if log:
            print(reply)
        return reply


class ChatWithStorage(BaseChat):

    def __init__(self, *args, session_id=None, system_message=''):
        super().__init__(*args)
        self.session_id = session_id
        self.system_message = system_message
        self.message_history = SQLChatMessageHistory(
            session_id=session_id, connection_string="sqlite:///chat.db"
        ) if session_id else ChatMessageHistory()
        self.chain = self.create_chain()

        if self.message_history.messages:
            print(self.message_history)
        elif system_message:
            self.message_history.add_message(("system", system_message))

    def create_chain(self):
        output_parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("user", "{question}"),
        ])
        chain = prompt | self.llm | output_parser
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.message_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        return chain_with_history

    def query(self, question, log=False):
        config = {"configurable": {"session_id": self.session_id}}
        reply = ''
        for chunk in self.chain.stream({"question": question}, config=config):
            if log:
                print(chunk, end="", flush=True)
            reply += chunk
        if log:
            print('')
        return reply


class ChatWithSummarizedHistory(BaseChat):

    def __init__(self, *args, system_message=''):
        super().__init__(*args)
        self.system_message = system_message
        self.chat_history = ChatMessageHistory()
        self.chain = self.create_chain()

    def create_chain(self):
        output_parser = StrOutputParser()
        messages = (
            [("system", self.system_message)]
            if self.system_message else []
        )
        messages.extend([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        chain_with_message_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        chain_with_summarization = (
            RunnablePassthrough.assign(messages_summarized=self.summarize_messages)
            | chain_with_message_history
            | output_parser
        )
        return chain_with_summarization

    def summarize_messages(self, chain_input):
        stored_messages = self.chat_history.messages
        if len(stored_messages) == 0:
            return False
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
                ),
            ]
        )
        summarization_chain = summarization_prompt | self.llm
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
        self.chat_history.clear()
        self.chat_history.add_message(summary_message)
        return True

    def query(self, question, log=False):
        reply = ''
        stream = self.chain.stream(
            {"input": question},
            {"configurable": {"session_id": "unused"}},
        )
        for chunk in stream:
            if log:
                print(chunk, end="", flush=True)
            reply += chunk
        if log:
            print('')
        return reply
