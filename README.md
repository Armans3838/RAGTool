# RAGTool
This is an attempt to build an appliction that allows users to use RAG in addition to an LLM in order to quickly query and analyze PDFs and other documents.

## Introduction
RAG or Retrieval Augmented Generation is a model that uses a retriever to find relevant information and then uses a generator to generate a response. This is useful for finding information in large documents or databases or for summarizing information that an LLM may not have seen before. RAG has been shown to provide an excellent alternative to retraining LLMs on new data or even fine-tuning them.
![RAG Basics](RAGpic.png "RAG Basics")
**RAG Basics**

### Data Ingestion
The first thing we need to do is ingest our data. Using LLamaIndex we performed basic data ingestion with the `SimpleDirectorrReader` class. Once our document(s) were loaded we could then create our Index.

### Embedding Generation
Creating or deciding on an embedding model is a crucial step in the process. The embedding model is what is used to generate the embeddings for the documents. In our case, we used the `Alibaba-NLP/gte-Qwen2-1.5B-instruct` model. This model was chosen because it is a large model that has been trained on a variety of data and has been shown to perform well in a variety of tasks. The model is also available on the HuggingFace model hub. This embedding model has 1.5 billion parameters and performs incredibly well without any additional node generation tequniques.

The Index is what is responsible for storing the embeddings of our documents. The following image shows the basic structure of generating embeddings.
![Embedding Basics](embeddingBasics.png "Embedding Basics")
**Embedding Basics**

As you can see, the knowledge graph has "groups" of data that are then used to generate embeddings. These embeddings are then stored in the Index. The Index itself can be stored in several different ways- LlamaIndex for example supports a plethora of storage options ranging from Open Source databases to cloud storage. Something to note about the colors found in the knowledge graph is that they represent the different "groupings" of data. These groups are known as Nodes and are created based on the embedding model or with advanced node generation techniques.

### Vector Store Index
In our RAGTool, we made use of the LLamaIndex `VectorStoreIndex` class to store our embeddings, and also then stored the generated vectors to a director allowing us to quickly load the embeddings for use in our RAG model. Bellow is the general setup of the Vector Store.
![Vector Store](vectorStorePic.png "Vector Store")
**Vector Store** - Basic Storage of Nodes and their respective embeddings.

![Vector Store Query](vectorStoreQueryPic.png "Vector Store Query")
**Vector Store Query**
As you can see, when we query our RAG, we are querying the vector store for the embeddings of the documents we are interested in.



## Installation
Use the requirements.txt file to install the necessary packages.

```bash
pip install -r requirements.txt
```


## Usage
To run the application, use the following command:

```python
python main.py <embedding_model> <embedding_path> <document>
```

Where `<embedding_model>` is the name of the embedding model, `<embedding_path>` is the path to the persisted embedding model, and `<document>` is the path to the document you want to analyze.



## Sources
[LlamaIndex](https://docs.llamaindex.ai/en/stable/)
[HuggingFace](https://huggingface.co)
[MTEB LeaderBoard](https://huggingface.co/spaces/mteb/leaderboard)
[Embedding Model](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)
[Semantic Double Merge Performance](https://bitpeak.com/chunking-methods-in-rag-methods-comparison/)