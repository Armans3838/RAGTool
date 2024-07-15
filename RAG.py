import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.lmstudio import LMStudio
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import (SentenceSplitter, SemanticSplitterNodeParser, SemanticDoubleMergingSplitterNodeParser, LanguageConfig)
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (messages_to_prompt, completion_to_prompt)
from llama_cpp import Llama
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer



# global variables
data = None
embedding_model = None



class RAG:
    def __init__(self, data_path, model):
        global data
        global embedding_model
        data = data_path
        embedding_model = model

        RAG.__set_global_settingsLMStudio()
        # RAG.__set_global_settingsLlamaCPP()



    def __is_folder_empty(path):
        return not os.listdir(path)



    # Load the documents from the data path
    def load_documents(self):
        loaded_documents = SimpleDirectoryReader(data).load_data(show_progress=True) #, num_workers=4)
        return loaded_documents



    # Set the global settings for LlamaIndex
    def __set_global_settingsLMStudio():
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        Settings.llm = LMStudio(
            model_name = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            base_url="http://localhost:1234/v1",
            temperature=0.7, # changing this value will change the randomness of the responses with 0 being deterministic and 1 being completely random
            request_timeout=60
        )


    # Working on allowing this program to be run with LlamaCPP_Python as opposed to LlamaStudio alone. Not done yet.
    # def __set_global_settingsLlamaCPP():
    #     Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model, trust_remote_code=True)

    #     # llm = Llama.from_pretrained(
    #     #     repo_id="lmstudio-community/Meta-Llama-3-8B-Instruct-BPE-fix-GGUF",
    #     #     filename="Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    #     #     verbose=False
    #     # )

    #     set_global_tokenizer(AutoTokenizer.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5").encode)

    #     Settings.llm = LlamaCPP(
    #         # You can pass in the URL to a GGML model to download it automatically
    #         model_url="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/tree/main",
    #         # optionally, you can set the path to a pre-downloaded model instead of model_url
    #         model_path=None,
    #         temperature=0.7,
    #         max_new_tokens=256,
    #         # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    #         context_window=3900,
    #         # kwargs to pass to __call__()
    #         generate_kwargs={},
    #         # kwargs to pass to __init__()
    #         # set to at least 1 to use GPU
    #         model_kwargs={"n_gpu_layers": 8},
    #         # transform inputs into Llama2 format
    #         messages_to_prompt=messages_to_prompt,
    #         completion_to_prompt=completion_to_prompt,
    #         verbose=True,
    #         # request_timeout=60
    #     )



    # This function creates the index for a given embedding model and stores it in persistent storage
    # The function provides no control on how the chunks are created and therefore how the embeddings are created
    def plainEmbedding(self, embedding_path, loaded_documents):
        print("Creating index")
        vector_index = VectorStoreIndex.from_documents(loaded_documents, show_progress=True)
        print("Creating embeddings and storing them in persistent storage...\n")
        vector_index.storage_context.persist(persist_dir=embedding_path)
        print("Index has been created and loaded. Ready for deployment!\n\n\n")



    # This function creates the index for a given embedding model and stores it in persistent storage
    # The function uses the SemanticSplitterNodeParser to create the chunks and therefore the embeddings
    # The SemanticSplitterNodeParser uses the spacy model to split the documents into chunks based on their semantic similarity
    # A threshold is used to determine the similarity between two sentences and places them inside of the same chunk
    # A different threshold is used to determine if the next sentence should be placed in a new chunk or the previous one
    # Finally, a threshold is used to determine if two chunks should be merged together on the "second pass"
    def SemDoubleMerge(self, embedding_path, loaded_documents):
        print("Defining the node parser based on SemanticDoubleMergingSplitterNodeParser")
        config = LanguageConfig(language="english", spacy_model="en_core_web_md")
        splitter = SemanticDoubleMergingSplitterNodeParser(
            language_config=config,
            initial_threshold=0.7, # threshold for making a new chunk
            appending_threshold=0.5, # threshold for appending a sentence to the previous chunk
            merging_threshold=0.65, # threshold for merging two chunks together
        )
        print("Creating nodes...\n")
        nodes = splitter.get_nodes_from_documents(loaded_documents)

        print("Creating embeddings and storing them in persistent storage...")
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.storage_context.persist(persist_dir=embedding_path)

        print("Index has been created and loaded. Ready for deployment.\n\n\n")



    # A function to retrieve the index from persistent storage
    def embeddingRetrieval(self, embedding_path):
        print("Index has already been created- loading index from persistent storage\n\n\n")
        storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
        vector_index = load_index_from_storage(storage_context)

        return vector_index



    # A function to create the chat engine in a while loop
    def chatEngine(self, vector_index):
        print("Creating chat engine...\n")
        chat_engine = vector_index.as_chat_engine(chat_mode="condense_plus_context", memory=ChatMemoryBuffer.from_defaults(token_limit=3900))
        while True:
            user_response = input("Enter your question: ")  # Prompt the user for input
            print("thinking...\n")
            response = chat_engine.chat(user_response)
            print(response)