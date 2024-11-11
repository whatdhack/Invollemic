from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama

import os
import torch
from transformers import pipeline, AutoTokenizer

from tensorrt_llm import LLM, SamplingParams, BuildConfig

import argparse
parser = argparse.ArgumentParser(description="LLamap CLI.")
parser.add_argument("--server", help="trt, hf, ollama", default="hf")
args  = parser.parse_args()



def trt_llm():
    build_config = BuildConfig()
    build_config.plugin_config.gemm_plugin = 'auto'
    build_config.plugin_config.context_fmha =False
    
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", build_config=build_config)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def hf_pipe():
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])

def ollama_llm():
    # ./ollama/bin/ollama serve
    # ./ollama/bin/ollama run llama3.2:1b
    locally_run = Ollama(model="llama3.2:1b", request_timeout=360.0)
    return locally_run

def hf_llm():
    model_name = "meta-llama/Llama-3.2-1B"
    model_config = {
        # Pydantic 2 sets `model_` prefix as protected namespace and raises a
        # warning when this model is loaded. Set this option to disable the warning.
        "protected_namespaces": (),
    }
    locally_run_model = HuggingFaceLLM(model_name=model_name, 
                                 tokenizer=AutoTokenizer.from_pretrained(model_name),
                                 )
    embedder_model_name = "sentence-transformers/all-mpnet-base-v2"
    locally_run_embed =  HuggingFaceEmbedding(model_name=embedder_model_name)
    return locally_run_model, locally_run_embed

llm = hf_llm
if args.server == "trt":
    llm = trt_llm
elif args.server == "ollama":
    llm = ollama_llm
    
    
documents = SimpleDirectoryReader("data").load_data()

Settings.llm,Settings.embed_model = llm()


index = VectorStoreIndex.from_documents(
    documents,
    )

query_engine = index.as_query_engine()
response = query_engine.query("Summarize the invoices")
print(response)

