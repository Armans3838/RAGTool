import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

from llama_index.llms.lmstudio import LMStudio
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Fix asyncio bug
nest_asyncio.apply()


documents = SimpleDirectoryReader("data").load_data()


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# lmstudio
Settings.llm = LMStudio(
    model_name="TheBloke/Llama-2-7B-Chat-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0.7
)

# messages = [
#     ChatMessage(
#         role=MessageRole.SYSTEM,
#         content="You an expert AI assistant. Help User with their queries.",
#     ),
#     ChatMessage(
#         role=MessageRole.USER,
#         content="How do I add numbers together?",
#     ),
# ]

# Create a new index
index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine()

response = query_engine.query("Who died January 15, 2014, and why did they die?")
print((response))

# index = VectorStoreIndex.from_documents(
#     documents,
# )