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

# current time for log name
now = datetime.now()
t = now.strftime("%H:%M:%S")

# If using complex names such as urls, considere making another
base_data = "https://www.youtube.com/watch?v=tkH2-_jMCSk"
inputArray = ["https://www.youtube.com/watch?v=wTBSGgbIvsY" , # meditation
    "https://www.youtube.com/watch?v=EQ3GjpGq5Y8" , # Sauna & heat
    "https://www.youtube.com/watch?v=XcvhERcZpWw" , # Nutrients cold longevity
    "https://www.youtube.com/watch?v=pq6WHJzOkno" , # Cold
    "https://www.youtube.com/watch?v=DTCmprPCDqc" , # Nutrition, Exercise, Hormones, Vitality
    "https://www.youtube.com/watch?v=K4Ze-Sp6aUE" , # Eating
    "https://www.youtube.com/watch?v=9tRohh0gErM" , # Fasting
    "https://www.youtube.com/watch?v=yaWVflQolmM" , # fasting 2
    "https://www.youtube.com/watch?v=ulHrUVV3Kq4", # stress
    "https://www.youtube.com/watch?v=UIy-WQCZd4M", # Fitness & longitity
    "https://www.youtube.com/watch?v=oNkDA2F7CjM", # endurance & fat loss
    "https://www.youtube.com/watch?v=tLS6t3FVOTI", # supplements
    "https://www.youtube.com/watch?v=UNCwdFxPtE8", # Exercise
    "https://www.youtube.com/watch?v=T65RDBiB5Hs", # brain chem
    "https://www.youtube.com/watch?v=szqPAPKE5tQ", # improved memory
    "https://www.youtube.com/watch?v=azb3Ih68awQ", # Performance
    "https://www.youtube.com/watch?v=17O5mgXZ9ZU", # Eating disorders
    "https://www.youtube.com/watch?v=nwSkFq4tyC0", # learning & metabolism
    "https://www.youtube.com/watch?v=xaE9XyMMAHY", # endurance & strenght
    "https://www.youtube.com/watch?v=E7W4OQfJWdw" # Brains & performance
]


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
            doc_part.extra_info = {"theme": inputdata}
        except:
            print("Data problem with input, skipping")

        all_docs.append(doc_part)

# ------------------
### INDEXES
# ------------------

base_index = GPTSimpleVectorIndex.load_from_disk("huberman_graph_03_xl", service_context=service_context)

for doc in all_docs:
    try:
        base_index.insert(doc)
    except:
        print(f"Problem with generating index")


graph = ComposableGraph.from_indices(GPTListIndex, [base_index], index_summaries=["Health information podcasts"], service_context=service_context)
graph.save_to_disk("Huberman_graph_01_insert")
