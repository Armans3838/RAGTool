import nest_asyncio
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.lmstudio import LMStudio
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Fix asyncio bug
nest_asyncio.apply()






# simple check to see if the folder is empty or not. used to determine if we need to create the index or not
def is_folder_empty(path):
    return not os.listdir(path)

folder_path = "persist_dir_Trig"




# The document represents the data that we want to index. It is broken down into nodes which represent
# a "chunk" of the source Document. Both the document and the nodes contain metadata and relationships.
# LamaIndex provides the SimpleDirectoryReader class acting as a wrapper for several different data sources.

# documents = SimpleDirectoryReader("data").load_data()
documents = SimpleDirectoryReader("trig").load_data(show_progress=True) #, num_workers=4)


# Setting the embedding model to be used in the index
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# lmstudio
Settings.llm = LMStudio(
    model_name="bartowski/Phi-3-Context-Obedient-RAG-GGUF",
    # model_name = "TheBloke/Llama-2-7B-Chat-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0.7, # changing this value will change the randomness of the responses with 0 being deterministic and 1 being completely random
    request_timeout=60
)



# The Index is the core of LlamaIndex. It's what allows RAG to function as it does. The index is created from the documents
# which are then used to build Query Engines and Chat Engines. Indexes sotre data in Nodes and then expose a 
# Retriever.


# creating our index and then recalling that index from persistent memorry if it's not empty
if(is_folder_empty(folder_path)):
    print("Creating index")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=folder_path)
else:
    print("Index has already been created- loading index from persistent storage")
    storage_context = StorageContext.from_defaults(persist_dir=folder_path)
    index = load_index_from_storage(storage_context)



memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
query_engine = index.as_query_engine()
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", memory=memory)


while True:
    user_response = input("Enter your question: ")  # Prompt the user for input
    # response = query_engine.query(user_response) # Use the user's response in chat_engine.chat()
    print("thinking...")
    response = chat_engine.chat(user_response)
    print(response)