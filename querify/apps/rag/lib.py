from abc import (
    ABC,
    abstractmethod,
)
from operator import itemgetter

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    get_buffer_string,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStoreRetriever

from utils import clean_prompt


default_document_prompt = PromptTemplate.from_template(template="{page_content}")
def combine_documents(docs, document_prompt=default_document_prompt, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def create_simple_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(clean_prompt("""
        Answer the following question based only on the following context:
        {context}

        Question: {question}
    """))
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def query_simple_chain(chain, msg, log=False):
    reply = chain.invoke(msg)
    if log:
        print(reply)
    return reply


def create_advanced_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(clean_prompt("""
        Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}
    """))
    # Note! This passes ALL documents; make sure they fit within the context window of the LLM in use.
    document_chain = create_stuff_documents_chain(llm, prompt)
    # construct a chain that passes the user inquiry to the retriever to fetch relevant documents
    # The retrieved documents (and original inputs) are then passed to an LLM to generate a response
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def query_advanced_chain(chain, msg, log=False):
    response = chain.invoke({"input": msg})
    if log:
        print(response["answer"])
        print('context:', len(response['context']))
    return response


class BaseRAGChat(ABC):

    def __init__(self, llm: BaseChatModel, retriever: VectorStoreRetriever):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def query(self, question: str):
        pass


class Chat(BaseRAGChat):

    def __init__(self, *args):
        super().__init__(*args)
        self.chat_history = []
        self.retriever_chain, self.retrieval_chain = (
            self.create_history_aware_retrieval_chain(self.retriever)
        )

    def create_history_aware_retrieval_chain(self, retriever):
        # A prompt to be passed into an LLM to generate the search query
        retriever_chain_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        # This chain takes in conversation history and then uses that to generate a search query which is passed to the underlying retriever.
        retriever_chain = create_history_aware_retriever(self.llm, retriever, retriever_chain_prompt)
        document_chain_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(self.llm, document_chain_prompt)
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        return retriever_chain, retrieval_chain

    def query(self, msg, log=False):
        response = self.retrieval_chain.invoke({
            "chat_history": self.chat_history,
            "input": msg
        })
        self.chat_history.extend([
            HumanMessage(content=msg),
            AIMessage(content=response['answer'])
        ])
        if log:
            print(response['answer'])
            print('context:', len(response['context']))
        return response

    def get_documents(self, msg):
        return self.retriever_chain.invoke({
            "chat_history": self.chat_history,
            "input": msg
        })


class ChatWithHistory(BaseRAGChat):

    def __init__(self, *args):
        super().__init__(*args)
        self.chat_history = []
        self.chain = self.create_chain(self.retriever)

    def create_chain(self, retriever):
        condense_question_prompt = PromptTemplate.from_template(clean_prompt("""
            Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:
        """))
        answer_prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the following context:
            {context}

            Question: {question}
        """)
        standalone_question = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | condense_question_prompt
            | self.llm
            | StrOutputParser(),
        )
        context = {
            "context": itemgetter("standalone_question") | retriever | combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        return standalone_question | context | answer_prompt | self.llm

    def query(self, question, log=False):
        reply = self.chain.invoke(
            {
                "question": question,
                "chat_history": self.chat_history,
            }
        )
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=reply.content)
        ])
        if log:
            print(reply.content)
        return reply


class ChatWithMemory(BaseRAGChat):

    def __init__(self, *args):
        super().__init__(*args)
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
        self.chain = self.create_chain(self.retriever)

    def create_chain(self, retriever):
        condense_question_prompt = PromptTemplate.from_template(clean_prompt("""
            Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:
        """))
        answer_prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the following context:
            {context}

            Question: {question}
        """)
        # load memory
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )
        # generate standalone question
        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | condense_question_prompt
            | self.llm
            | StrOutputParser(),
        }
        # retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"],
        }
        # construct context input for the prompt
        context = {
            "context": lambda x: combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }
        # returns answers
        answer = {
            "answer": context | answer_prompt | self.llm,
            "docs": itemgetter("docs"),
        }
        return loaded_memory | standalone_question | retrieved_documents | answer

    def query(self, question, log=False):
        inputs = {"question": question}
        result = self.chain.invoke(inputs)
        # Memory doesn't save automatically
        self.memory.save_context(inputs, {"answer": result["answer"].content})
        if log:
            print(result["answer"].content)
            print('docs:', len(result["docs"]))
        return result

    def get_memory(self):
        return self.memory.load_memory_variables({})
