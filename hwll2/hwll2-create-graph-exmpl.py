from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
from llama_index import LangchainEmbedding, GPTListIndex, GPTTreeIndex
from llama_index.indices.composability import ComposableGraph
from llama_index import LLMPredictor, GPTSimpleVectorIndex, ServiceContext
from run_env import *
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from pathlib import Path
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from typing import Optional, List, Mapping, Any
import sys
from datetime import datetime

########################################
# Generate new graph from raw data
# Save to disk
########################################

# ------------------
# GLOBAL STATES
# ------------------
all_docs = [] # store Documents
all_indexes = [] # store indexes
all_summaries = []

LlamaArgs = {
    "model_path": path_to_model,
    "n_ctx": 2048,
}

# If using complex names such as urls, considere making another
base_data = "https://www.youtube.com/watch?v=tkH2-_jMCSk"
inputArray = ["https://www.youtube.com/watch?v=wTBSGgbIvsY"]

# ------------------
### INIT
# ------------------
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=LlamaCpp(**LlamaArgs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024, embed_model=embed_model)


# ------------------
### DATA LOADER
# ------------------
YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
loader = YoutubeTranscriptReader()

# Go over input data
base_doc = loader.load_data(ytlinks=[base_data])
for inputdata in inputArray:
    x_docs = loader.load_data(ytlinks=[inputdata]) # Doc

    # Put metadata into data
    for doc_part in x_docs:
        try:
            doc_part.extra_info = {"page": inputdata}
        except:
            print("Data problem with input, skipping")

        all_docs.append(doc_part)

# ------------------
### INDEXES
# ------------------

base_index = GPTSimpleVectorIndex.from_documents(base_doc, service_context=service_context)

for doc in all_docs:
    try:
        base_index.insert(doc)
    except:
        print(f"Problem with generating index")


graph = ComposableGraph.from_indices(GPTListIndex, [base_index], index_summaries=["Health information podcasts"], service_context=service_context)
graph.save_to_disk("huberman_graph_03_xl")
