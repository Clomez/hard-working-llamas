from langchain.llms import LlamaCpp
from run_env import *

########################################
# Start alpacha only. 
########################################

LlamaArgs = {
    "model_path": path_to_model,
    "n_ctx": 2048,
}

LlamaCpp(**LlamaArgs)