from llama_index import GPTListIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from langchain.llms.base import LLM
from llama_index import LLMPredictor, GPTSimpleVectorIndex, ServiceContext
from typing import Optional, List, Mapping, Any

from pathlib import Path
from llama_cpp import Llama
from langchain.llms.base import LLM
from run_env import *


class LlamaLLM(LLM):
    model_path: str
    llm: Llama

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"

    def __init__(self, model_path: str, **kwargs: Any):
        model_path = model_path
        llm = Llama(model_path=model_path)
        super().__init__(model_path=model_path, llm=llm, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.llm(prompt, stop=stop or [])
        return response["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

# load model, llama
llm_predictor = LLMPredictor(llm=LlamaLLM(model_path=path_to_model))

# service context generation
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512, embed_model=embed_model)

# load index
new_index = GPTListIndex.load_from_disk('index_list_emb.json')

# query with embed_model specified
response = new_index.query(
    "<query_text>", 
    mode="embedding", 
    verbose=True, 
    service_context=service_context
)
print(response)