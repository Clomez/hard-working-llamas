from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
from pathlib import Path
from llama_cpp import Llama
from langchain.chains.question_answering import load_qa_chain
from llama_index import LangchainEmbedding
from llama_index import LLMPredictor, GPTSimpleVectorIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

# Load array of wikipedia articles to llama model
# Using local llama model and huggingface embeding
# NO API KEY NEEDED

## LLama wrapper class
## TODO: into separate file u go
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



# Global shaits
inputArray = ['Berlin', 'Rome', 'Tokyo', 'Canberra', 'Santiago'] # Wiki articles

# Internal magic
index_set = {}
doc_set = {}
all_docs = []

# Loader
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()

# load in HF embedding model from langchain
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

# load model, llama
llm_predictor = LLMPredictor(llm=LlamaLLM(model_path="../toModel"))

# service context generation
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512, embed_model=embed_model)


# iterate data-loading
for i in inputArray:
    year_docs = loader.load_data(pages=[i])

    # insert year metadata into each year
    for d in year_docs:
        d.extra_info = {"page": i}
    doc_set[i] = year_docs
    all_docs.extend(year_docs)

# initialize simple vector indices + global vector index
for doc_id in inputArray:
    cur_index = GPTSimpleVectorIndex.from_documents(doc_set[doc_id], service_context=service_context)
    index_set[doc_id] = cur_index
    cur_index.save_to_disk(f'index_{doc_id}.json')