# hard-working-llamas
- llama-index-HF-chainin-tool
- Example motivation code for making llamas work for you!

1. Generate data from source (Wiki, Youtube, Reddit)
2. Run model with generated JSON extension.

NO API KEY NEEDED

requirements:
- llama model
- python3 & pip
- Huggingface embeded model - downloadable via pip

# Usage
1. Create file "run_env.py" containing path to model
example: path_to_model="path/to/model/model.bin"
2. Install needed packages
3. run script 'python3 hard-working-llama.py'

1. Insert question parameters into hwll-use-json.py
2. Insert JSON file into hwll-use-json.py
2. Run hwll-use-json.py

# Info / content
Example motivation code for making llamas work for you!

1. hard-working-llama - Original: 
Dataloader: Wikipedia - array
Model: Llama 7B (tested)
Embeded model: Embeded Huggingface

Save indexes to disk

2. hwll-youtube.py
Dataloader: Youtube - array
Model: Llama 7B (tested)
Embeded model: Embeded Huggingface

Save indexes to disk

3. hwll-use-json.py
Use JSON files to answer questions