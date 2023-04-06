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
### QUERY CONFIG
# ------------------

def query_configs():
    return [
        {
            "index_struct_type": "tree",
            "query_mode": "default",
            "query_kwargs": {
                "child_branch_factor": 2
            }
        }
    ]

def prompt():
    a_name = "Havel"

    personality = f"""
You are {a_name}. {a_name} is a scientist and assistant. If {a_name} dosent know the answer, he will not lie. {a_name} always tells you, how he got the answer. {a_name} always gives you sources for answer. {a_name} gives very long answers. {a_name} gives very detailed answers.
    """

    question = f"""
{a_name} summarize the effects of deliberate heat exposure
    """

    return personality + question

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
inputArray = ["https://www.youtube.com/watch?v=tkH2-_jMCSk",
    "https://www.youtube.com/watch?v=wTBSGgbIvsY" , # meditation
    "https://www.youtube.com/watch?v=EQ3GjpGq5Y8" , # Sauna & heat
    "https://www.youtube.com/watch?v=XcvhERcZpWw" , # Nutrients cold longevity
    "https://www.youtube.com/watch?v=pq6WHJzOkno" , # Cold
    "https://www.youtube.com/watch?v=x7qbJeRxWGw" , # Metabolism & nutrients
    "https://www.youtube.com/watch?v=DTCmprPCDqc" , # Nutrition, Exercise, Hormones, Vitality
    "https://www.youtube.com/watch?v=K4Ze-Sp6aUE" , # Eating
    "https://www.youtube.com/watch?v=9tRohh0gErM" , # Fasting
    "https://www.youtube.com/watch?v=yaWVflQolmM" , # fasting 2
]

# ------------------
### INIT
# ------------------
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=LlamaCpp(**LlamaArgs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512, embed_model=embed_model)


# ------------------
### DATA LOADER
# ------------------
YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
loader = YoutubeTranscriptReader()

# Go over input data
for inputdata in inputArray:
    x_docs = loader.load_data(ytlinks=[inputdata])

    # Put metadata into data
    for doc_part in x_docs:
        try:
            doc_part.extra_info = {"page": inputdata}
        except:
            print("Data problem with input, skipping")

    # Loop data to list, print error if no data
    try:
        all_docs.append(x_docs)

    except:
        print("Data problem with input, skipping")


# ------------------
### INDEXES
# ------------------

# Loop Document List to indexes
for i in all_docs:
    index = GPTSimpleVectorIndex.from_documents(i, service_context=service_context)
    try:
        all_indexes.append(index)
    except:
        print(f"Problem with generating index {i}")


# repsonse = all_indexes[0].query("why should you stretch?", verbose=True)

# ------------------
### RUN QUERY
# ------------------
# Summaries lenght MUST match input data lenght - all entries need summary
graph = ComposableGraph.from_indices(GPTListIndex, all_indexes, index_summaries=["stretching" "meditation", "Sauna & heat", "Nutrients cold longevity","Cold","Metabolism & nutrients","Nutrition, Exercise, Hormones, Vitality", "Eating", "fasting", "fasting 2"], service_context=service_context)
graph.save_to_disk("huberman_index_01")

response = graph.query(str(prompt()), query_configs=query_configs(), service_context=service_context)


# ------------------
### RESULT HANDLING
# ------------------
print(response)
