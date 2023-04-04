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


# Global shaits & Internal magic
index_set = {} # not used
doc_set = {} # Final doc
all_docs = [] # ?
GPindex = {}

LlamaArgs = {
    "model_path": path_to_model,
    "n_ctx": 2048,
#    "n_threads": 14
#    "use_mlock": True
}

inputArray = ['https://www.youtube.com/watch?v=EQ3GjpGq5Y8',
        'https://www.youtube.com/watch?v=XcvhERcZpWw']

print("config " + "Path=" + path_to_model)

# PDFReader = download_loader("PDFReader")
YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")

loader = YoutubeTranscriptReader()
# loader = WikipediaReader()
# loader = PDFReader()

# documents = loader.load_data(file=Path('./llama.pdf'))
# llama_emb.embed_documents(["asd"])
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
# llama_emb = LlamaCppEmbeddings(**LlamaArgs)
# embedding = LangchainEmbedding(llama_emb)


# load model, llama
llm_predictor = LLMPredictor(llm=LlamaCpp(**LlamaArgs))

# service context generation
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024, embed_model=embed_model)

# WikipediaReader = download_loader("WikipediaReader")
documents = loader.load_data(ytlinks=[inputArray[0]])
# x_docs = loader.load_data(pages=['Berlin'])

documents2 = loader.load_data(ytlinks=[inputArray[1]])
# x_docs = loader.load_data(pages=['Berlin'])
cur_index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
# cur_index = GPTSimpleVectorIndex.load_from_string(txt, service_context=service_context)

cur_index2 = GPTSimpleVectorIndex.from_documents(documents2, service_context=service_context)
# cur_index = GPTSimpleVectorIndex.load_from_string(txt, service_context=service_context)
# using JSON dictionaries
query_configs = [
    {
        "index_struct_type": "tree",
        "query_mode": "summarize",
        "query_kwargs": {
            "child_branch_factor": 2
        }
    }
]
# cur_index.query("Summary", mode="default")

graph = ComposableGraph.from_indices(GPTListIndex, [cur_index, cur_index2], index_summaries=["heat exposure", "health"], service_context=service_context)
response = graph.query("Summarize the text given", query_configs=query_configs, service_context=service_context)

print(response)
