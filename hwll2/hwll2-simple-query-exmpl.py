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


#################################
# Query existing graph from disk
#################################

def writeResultToFile(f, response):
    try:
        wr = str(response) + "\n"
        f.write(wr)
    except:
        raise Exception("Cant write to file")

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
    a_name = "Dr. Havel"

    personality = f"""
You are {a_name}. 
    """

    question = f"""
{a_name}, give me a long answer about the benefits of fasting, according to Dr. Huberman.
    """

    return personality + question

LlamaArgs = {
    "model_path": path_to_model,
    "n_ctx": 2048,
}

f = open("log", "a")
writeResultToFile(f, "------------ NEW QUERY ------------")
writeResultToFile(f, "Prompt: " + prompt())

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=LlamaCpp(**LlamaArgs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024, embed_model=embed_model)

index = ComposableGraph.load_from_disk("./Huberman_graph_01_insert", service_context=service_context)
graph = ComposableGraph.from_indices(GPTListIndex, [index], index_summaries=["Health information podcasts"], service_context=service_context)

response = graph.query(str(prompt()), query_configs=query_configs(), service_context=service_context)

print(response)
writeResultToFile(f, response)
writeResultToFile(f, "------------ QUERY END ------------")
f.close()
