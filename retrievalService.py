import os 
import git 
import shutil
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub, Replicate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory



load_dotenv()


class RetrievalService:
    def __init__(self, github_link):
        self.allowed_extensions = ['.py',  '.md']
        self.link = github_link
        self.repo_name = self.link.split('/')[-1]
        self.pathway = self.repo_name.split('.')[0]
        self.model = "HuggingFaceH4/zephyr-7b-beta" 
        self.root = self.repo_name
        self.docs = []
        self.memory = ConversationBufferWindowMemory(
            k=1,
            memory_key='chat_history', return_messages=True
            )
        self.persist_directory='db'


        self.embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. If possible give code snippets with the answers. But get the code from the context provided below.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else. If the answer of the question is not in the context, just say "I don't know". Please do not make up an answer or try to give it using your own knowledge.
        Helpful answer:
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])  # Creating a prompt template object


    def getting_repo(self):
        print("Getting repo")
        if not os.path.exists(self.pathway):
            git.Repo.clone_from(self.link, self.pathway)


    def get_docs(self):
        print("Getting docs")

        self.docs = []
        for directory, dirnames, filenames in os.walk(self.root):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in self.allowed_extensions:
                    try: 
                        loader = TextLoader(os.path.join(directory, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e: 
                        pass
        

    def embedding(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        self.texts = self.text_splitter.split_documents(self.docs)
        self.vectordb = Chroma.from_documents(documents=self.texts, embedding=self.embeddings, persist_directory=self.persist_directory)
        self.vectordb.persist()

        # self.delete_directory()

    def select_model(self, model):
        if (model == "llama70B"):
            self.llm =  Replicate(
                model="meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
                model_kwargs={"temperature": 0.1, "max_length": 1500, "top_p": 1},
            )
            print("llama70B")
        elif (model == "Mixtral 7x8"):
            self.llm = Replicate(
                model="mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10",
                model_kwargs={"temperature": 0.1, "max_length": 1500, "top_p": 1},
            )
            print("Mixtral 7x8")
        elif (model == "zephyr7b"):
            self.llm = HuggingFaceHub(
                repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.1}
            )
            print("zephyr7b")


    def delete_directory(self):
        path = self.root
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)

    
    def retrieval(self):
   
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = 2
        self.vectordb = Chroma(persist_directory=self.persist_directory, 
            embedding_function=self.embeddings)
        self.vector_retriever = self.vectordb.as_retriever(search_kwargs={"k": 2})

        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.vector_retriever],
                                       weights=[0.5, 0.5])


    def conversation(self, model):
        self.select_model(model)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.ensemble_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
            )  
        return conversation_chain

