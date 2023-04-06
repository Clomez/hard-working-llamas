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
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {
                "child_branch_factor": 2
            }
        }
    ]

def prompt():
    a_name = "Havel"

    personality = f"""
You are {a_name}
{a_name} is a scientist and assistant.
{a_name} always does his best to answer questions honestly.
if {a_name} dosent know the answer, he will not lie.
{a_name} always tells you, how he got the answer.
{a_name} always gives you sources for answer.
{a_name} uses neutral language.
{a_name} gives very long answers.
{a_name} gives very detailed answers.
    """

    question = f"""
{a_name} tell me about ancient history of Rome?
    """

    return personality + question

# ------------------
### UTIL
# ------------------

def writeResultToFile(response):
    try:
        f.write(response)
        f.close()
    except:
        raise Exception("Cant write to file")

def try_writing():
    try:
        f = open(f"log-{t}", "a")
        return f
    except:
        raise Exception("Cant write to file")

# ------------------
# GLOBAL STATES
# ------------------
all_docs = [] # store Documents
all_indexes = [] # store indexes

LlamaArgs = {
    "model_path": path_to_model,
    "n_ctx": 2048,
}

# current time for log name
now = datetime.now()
t = now.strftime("%H:%M:%S")
f = try_writing()

# If using complex names such as urls, considere making another
inputArray = ['Berlin', 'Rome', 'Tokyo', 'Canberra', 'Santiago']

print(str(prompt()))


# ------------------
### INIT
# ------------------
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=LlamaCpp(**LlamaArgs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024, embed_model=embed_model)


# ------------------
### DATA LOADER
# ------------------
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()

# Go over input data
for inputdata in inputArray:
    x_docs = loader.load_data(pages=[inputdata])

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




# ------------------
### RUN QUERY
# ------------------
# Summaries lenght MUST match input data lenght - all entries need summary
graph = ComposableGraph.from_indices(GPTListIndex, all_indexes, index_summaries=inputArray, service_context=service_context)
response = graph.query(str(prompt()), query_configs=query_configs(), service_context=service_context)


# ------------------
### RESULT HANDLING
# ------------------
print(response)
writeResultToFile(response)
