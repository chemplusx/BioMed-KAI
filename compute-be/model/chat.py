import random
import json

import torch
from .llama_if import LlamaCppModel
# from schemas.rag import KGRag

from pathlib import Path

import logging

logger = logging.getLogger('MEDAL-AI')

# from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = None
tokenizer = None
rag_pipeline = None

def load_model(model_name):
    global model, tokenizer, rag_pipeline

    if not model:
        model_name = model_name
        path = Path(model_name)
        # path = Path('D:\\workspace\\models\\meditron-7b-q8_0.gguf')
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob('*.gguf'))[0]

        logger.info(f"llama.cpp weights detected: \"{model_file}\"")

        model, tokenizer = LlamaCppModel.from_pretrained(model_file)
        # rag_pipeline = KGRag("neo4j", model)
        # rag_pipeline.initialise_vector_indexes()

    return model, tokenizer

def unload_model():
    global model, tokenizer
    del model
    del tokenizer
    model = None
    tokenizer = None

bot_name = "MEDAL-AI"

def get_response(sentence, callback):
    global model, tokenizer, rag_pipeline
    if not model:
        ret_val = f"{bot_name}: No model loaded. Please load a model first."
        return ret_val
    
    response = model.generate2(sentence, {}, callback)

    if response:
        ret_val = f"{response}"
        return ret_val
    print(f"I do not understand...", sentence)
    return "Null"
