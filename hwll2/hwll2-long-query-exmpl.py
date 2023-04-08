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

##################################################
# Example query
# Long version
# Using question templates
# 
# Extensive Questions (template)
#  - Theme - overview of subject
#  - Summary - summary about subject
#  - Benefits - possible benefits of the action
#  - pratical use - tips on how to impliment
#
# Simple Question
#  - Title - shown with the results
#  - Question - question asked
##################################################

debug = True # Write prompts to file for review
debug_logger = False # 
writer_type = "FILE_MD"


def logger(msg):
    if(debug_logger):
        print(msg)

def write_md(f, response):
    try:
        wr = str(response) + "\n"
        f.write(wr)
    except:
        raise Exception("Cant write to file")

def writter(f, msg):
    if(writer_type == "FILE_MD"):
        write_md(f, msg)

def writterArr(f, arr):
    for i in arr:
        write_md(f, str(i))

def query_configs():
    return [
        {
            "index_struct_type": "dict",
            "query_mode": "recursive",
            "num_output": 1024,
            "query_kwargs": {
                "child_branch_factor": 3
            }
        }
    ]

def replace_themes(str, theme):
    return str.replace("--[THEME]--", theme)

def make_extra_query(item, question):
    res = graph.query(str(question), query_configs=query_configs(), service_context=service_context)
    logger("Extra Question: " + str(sum))
    logger("Answer: " + str(res))
    writter(f, "### " + str(question))
    writter(f, res)
    item["responses"].append(str(res)) 

def make_query(item, question):
    raw_str = item[question]
    sum = replace_themes(raw_str, item["theme"])
    res = graph.query(str(sum), query_configs=query_configs(), service_context=service_context)
    logger("Question: " + str(sum))
    logger("Answer: " + str(res))
    writter(f, "### " + str(sum))
    writter(f, res)
    item["responses"].append(str(res)) 

huberman_src = "Use data from Dr. Huberman and his guests. use up to 2000 words for the answer."

default_q_prompts = {
    "summary": f"""
Dr. Havel, Summarize all data you have about --[THEME]--. then tell about the results of 
studies about --[THEME]-- in detail. {huberman_src}
    """,
    "benefits": f"Dr. Havel, Can you tell about the benefits of --[THEME]-- in detail {huberman_src}",
    "pratical_tips": f"""
Dr. Havel, Can you tell pratical ways to get the benefits of  --[THEME]-- {huberman_src}
    """,
}

# Overview questions
questions = [
    {
        "theme": "deliberate heat exposure, for short time.", # Theme, such as "nutrition"
        "summary": default_q_prompts["summary"], # Give summary
        "benefits": default_q_prompts["benefits"], # Tell about benefits
        "pratical application": default_q_prompts["pratical_tips"], # pratical tips for use
        "title": "\n## Deliberate heat exposure.", # Title
        "responses": []
    },{
        "theme": "deliberate cold exposure, for short or long time.",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Cold exposure.",
        "responses": []
    },{
        "theme": "good nutrition diet versus bad nutrition diet",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Nutrition",
        "responses": []
    },{
        "theme": "regular sauna usage",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Sauna",
        "responses": []
    },{
        "theme": "Regular intermittent Fasting.",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Intermittent Fasting",
        "responses": []
    },{
        "theme": "Regular meditation.",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Meditation",
        "responses": []
    },{
        "theme": "Increased metabolism.",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Metabolism",
        "responses": []
    },{
        "theme": "Decresed metabolism.",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Metabolism",
        "responses": []
    },{
        "theme": "Stress reduction and stress.",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Stress reduction.",
        "responses": []
    },{
        "theme": "suppliments and drugs",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Suppliments.",
        "responses": []
    },{
        "theme": "Stretching",
        "summary": default_q_prompts["summary"],
        "benefits": default_q_prompts["benefits"],
        "pratical application": default_q_prompts["pratical_tips"],
        "title": "\n## Stretching.",
        "responses": []
    }
]

extra_questions = [
    {
        "question": f"Can you tell me about ketosis? {huberman_src}", # Give summary
        "title": "\n## About ketosis.", # Title
        "responses": []
    },
    {
        "question": f"What are the benefits of ketosis? {huberman_src}", # Give summary
        "title": "\n## Benefits of ketosis.", # Title
        "responses": []
    },
    {
        "question": f"Is stretching before sports good? {huberman_src}", # Give summary
        "title": "\n## Stretching before sports.", # Title
        "responses": []
    },
    {
        "question": f"What is the best way to increase mobility? {huberman_src}", # Give summary
        "title": "\n## Best way to increase mobility", # Title
        "responses": []
    },
    {
        "question": f"Can you tell me about the benefits of doing yoga? {huberman_src}", # Give summary
        "title": "\n## Benefits of yoga", # Title
        "responses": []
    },
    {
        "question": f"Are there any benefits in having regular ice baths? {huberman_src}", # Give summary
        "title": "\n## Ice baths.", # Title
        "responses": []
    },
    {
        "question": f"Are there any benefits in having regular cold showers? {huberman_src}", # Give summary
        "title": "\n## Cold showers.", # Title
        "responses": []
    },
    {
        "question": f"Give me examples of foods with good nutritional values. {huberman_src}", # Give summary
        "title": "\n## Good food for nutrition", # Title
        "responses": []
    },
    {
        "question": f"Can you give any advice on longevity? {huberman_src}", # Give summary
        "title": "\n## Longevity", # Title
        "responses": []
    },
    {
        "question": f"What are the top 3 exercises for increasing running endurance? {huberman_src}", # Give summary
        "title": "\n## Top exercises: Running endurance.", # Title
        "responses": []
    },
    {
        "question": f"What is the best way to increase oxygen intake and capacity? {huberman_src}", # Give summary
        "title": "\n## Best way to increase oxygen intake.", # Title
        "responses": []
    },
    {
        "question": f"Is running more than 20 kilometers or other long distances good for health? {huberman_src}", # Give summary
        "title": "\n## Is long distance running good for you", # Title
        "responses": []
    },
    {
        "question": f"How much running is too much? {huberman_src}", # Give summary
        "title": "\n## Best way to increase oxygen intake.", # Title
        "responses": []
    },
    {
        "question": f"what are the best exercises for bouldering, or rock climbing? {huberman_src}", # Give summary
        "title": "\n## Is long distance running good for you", # Title
        "responses": []
    },
    {
        "question": f"Can you tell about the benefits of bouldering, or rock climbing as exercise? {huberman_src}", # Give summary
        "title": "\n## How is bouldering beneficial as an exercise", # Title
        "responses": []
    },
    {
        "question": f"Is nicotine bad for health? {huberman_src}", # Give summary
        "title": "\n## How is bouldering beneficial as an exercise", # Title
        "responses": []
    },
]

def prompt():
    a_name = "Dr. Havel"

    personality = f"""
You are {a_name}. You like to write detailed but easy to understand health information for public.
Dont tell me who you are when answering.
All answers you give, are atleast 100 words long when possible.
    """

    question = f"""
{a_name}, What is the topic, that came up most often between Dr. Huberman and his guests.
    """

    return personality + question

LlamaArgs = {
    "model_path": path_to_model,
    "n_ctx": 2048,
}

date_time = datetime.now()
filen = str(date_time).replace(" ", "")
f = open(f"huberman{filen}.md", "a")

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=LlamaCpp(**LlamaArgs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024, embed_model=embed_model)

graph = ComposableGraph.load_from_disk("./huberman_graph_03_xl", service_context=service_context)

response = graph.query(str(prompt()), query_configs=query_configs(), service_context=service_context)
logger(response)

writter(f, "# Huberman Index 01\n")
writter(f, "## Most talked about topic in podcasts:")
writter(f, response)

for item in questions:
    writter(f, str(item["title"])) # Theme main title, such as 'Metabolism'
    make_query(item, "summary")
    make_query(item, "pratical application")
    make_query(item, "benefits")

writter(f, "## Extra single questions")
for item in extra_questions:
    make_extra_query(item, item["question"])

f.close()
