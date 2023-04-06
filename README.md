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

Two-stage-usage:
In order to run two-stage loaders you need two scripts
first one to load data into index and then those indexes to graph
Then you use second script to query the graph from previous step.

# Random tips
- Make sure your run_env.py is properly created!! (see example in this file)
- if using complicated inputArray, such as list of URL's, considere makinh another list
for summary texts, special characters may break the string, and then the whole run.


# Info / content
## Script types
Multi-stage:
Ran in two parts.
1. Download the data and make indexes
2. save indexes to disk

- Indexes and datasets can be reused from file
- Shorter runtime
- More complex

Single stage.
All in one
1. Select datasources and run
2. Save result to disk

- Simple

Original:
Archive - hard-working-llama:
Dataloader: Wikipedia - array
Model: Llama 7B (tested)
Embeded model: Embeded Huggingface

## Loaders
Wiki, youtube, pds. ...
- see loaders.md for loaders

## Run env
example run_env.py:

    path_to_model="path/to/model/model.bin"

    LlamaArgs = {
        "model_path": path_to_model,
        "n_ctx": 2048,
    }

