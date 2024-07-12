import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.lmstudio import LMStudio
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import (SentenceSplitter, SemanticSplitterNodeParser, SemanticDoubleMergingSplitterNodeParser, LanguageConfig)


# global variables
data = None
embedding_model = None



class RAG:
    def __init__(self, data_path, model):
        global data
        global embedding_model
        data = data_path
        embedding_model = model

        RAG.__set_global_settings()



    def __is_folder_empty(path):
        return not os.listdir(path)



    def load_documents(self):
        loaded_documents = SimpleDirectoryReader(data).load_data(show_progress=True) #, num_workers=4)
        return loaded_documents



    def __set_global_settings():
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        Settings.llm = LMStudio(
            model_name = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            base_url="http://localhost:1234/v1",
            temperature=0.7, # changing this value will change the randomness of the responses with 0 being deterministic and 1 being completely random
            request_timeout=60
        )



    def plainEmbedding(self, embedding_path, loaded_documents):
        print("Creating index")
        vector_index = VectorStoreIndex.from_documents(loaded_documents, show_progress=True)
        print("Creating embeddings and storing them in persistent storage...\n")
        vector_index.storage_context.persist(persist_dir=embedding_path)
        print("Index has been created and loaded. Ready for deployment!\n\n\n")



    def SemDoubleMerge(self, embedding_path, loaded_documents):
        print("Defining the node parser based on SemanticDoubleMergingSplitterNodeParser")
        config = LanguageConfig(language="english", spacy_model="en_core_web_md")
        splitter = SemanticDoubleMergingSplitterNodeParser(
            language_config=config,
            initial_threshold=0.7,
            appending_threshold=0.5,
            merging_threshold=0.65,
        )
        print("Creating nodes...\n")
        nodes = splitter.get_nodes_from_documents(loaded_documents)

        print("Creating embeddings and storing them in persistent storage...")
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.storage_context.persist(persist_dir=embedding_path)

        print("Index has been created and loaded. Ready for deployment.\n\n\n")



    def embeddingRetrieval(self, embedding_path):
        print("Index has already been created- loading index from persistent storage\n\n\n")
        storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
        vector_index = load_index_from_storage(storage_context)

        return vector_index



    def chatEngine(self, vector_index):
        print("Creating chat engine...\n")
        chat_engine = vector_index.as_chat_engine(chat_mode="condense_plus_context", memory=ChatMemoryBuffer.from_defaults(token_limit=3900))
        while True:
            user_response = input("Enter your question: ")  # Prompt the user for input
            print("thinking...\n")
            response = chat_engine.chat(user_response)
            print(response)