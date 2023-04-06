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


##########################
# Query Graph from disk
# - Write answer to disk
# - Child branch factor default: 1
##########################

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
    a_name = "Dr. Huberman"

    personality = f"""
You are {a_name}. 
{a_name} is a scientist and assistant.
If {a_name} dosent know the answer, he will express so.
he will not lie. {a_name} always tells you, how he got the answer.
{a_name} always gives you sources for answer.
{a_name} gives very long answers.
{a_name} gives very detailed answers.
{a_name} trusts people he has spoken to including his guests.
{a_name} trusts data from people he has spoken to.
    """

    question = f"""
{a_name}, write long and detailed summary text about how nutrition affects the brain, stress and mental health. Write about positive and negative effects.
Tell where you got the information in text.
Write text longer than 800 words.
    """

    return personality + question

LlamaArgs = {
    "model_path": path_to_model,
    "n_ctx": 2048,
}

f = open("log", "a")
writeResultToFile(f, "Prompt: " + prompt())

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=LlamaCpp(**LlamaArgs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512, embed_model=embed_model)

graph = ComposableGraph.load_from_disk("Huberman_graph_01_insert", service_context=service_context)
response = graph.query(str(prompt()), query_configs=query_configs(), service_context=service_context)

print(response)
writeResultToFile(f, response)
f.close()
