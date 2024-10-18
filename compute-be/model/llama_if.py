import re
from functools import partial

import numpy as np
import torch
# from langchain.embeddings.
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama
from langchain.chains import RetrievalQA
from huggingface_hub.hf_api import HfFolder
from .data_retreival import fetch_context

# from schemas.rag import KGRag

try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None

try:
    import llama_cpp_cuda_tensorcores
except:
    llama_cpp_cuda_tensorcores = None

token = "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"

HfFolder.save_token(token)

URI = "neo4j://192.168.0.115:7687"
AUTH = ("neo4j", "password")

def llama_cpp_lib():
    # if shared.args.cpu and llama_cpp is not None:
    #     return llama_cpp
    # elif shared.args.tensorcores and llama_cpp_cuda_tensorcores is not None:
    #     return llama_cpp_cuda_tensorcores
    if llama_cpp_cuda is not None:
        return llama_cpp_cuda
    else:
        return llama_cpp


def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits


def custom_token_ban_logits_processor(token_ids, input_ids, logits):
    for token_id in token_ids:
        logits[token_id] = -float('inf')

    return logits


class LlamaCppModel:
    models = {
            'llama-3.1': 'H:\\workspace\\NEXIS\\src\\models\\Meta-Llama-3-8B-Instruct.Q8_0.gguf',
            'llama-2': 'H:\\workspace\\NEXIS\\src\\models\\Meta-Llama-3-8B-Instruct.Q8_0.gguf'
        }
    model_name = 'llama-3.1'
    def __init__(self, model_name='llama-3.1'):
        self.model_name = model_name
        self.model = None
        self.initialized = False
        self.grammar_string = ''
        self.grammar = None
        self.rag_chain = None

    def __del__(self):
        del self.model

    @classmethod
    def get_models(cls):
        return cls.models

    @classmethod
    def load(self):

        Llama = llama_cpp_lib().Llama
        LlamaCache = llama_cpp_lib().LlamaCache

        result = self()
        cache_capacity = 0
        params = {
            'model_path': self.models[self.model_name],
            'n_ctx': 4096*16,
            'n_threads': 8,
            'n_threads_batch': None,
            'n_batch': 512,
            'use_mmap': True,
            'use_mlock': False,
            'mul_mat_q': True,
            'numa': False,
            'flash_attn': True,
            'n_gpu_layers': -1,
            'rope_freq_base': 500000,
            'rope_freq_scale': 1.0,
            'offload_kqv': True,
            'split_mode': 2
        }

        # Llama(**params)
        result.model = Llama(**params)
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        # result.initialise_vector_indexes()
        # result.setup_rag_pipeline()
        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result


    def encode(self, string):
        if type(string) is str:
            string = string.encode()

        return self.model.tokenize(string)

    def decode(self, ids, **kwargs):
        return self.model.detokenize(ids).decode('utf-8')

    def get_logits(self, tokens):
        self.model.reset()
        self.model.eval(tokens)
        logits = self.model._scores
        logits = np.expand_dims(logits, 0)  # batch dim is expected
        return torch.tensor(logits, dtype=torch.float32)

    def load_grammar(self, string):
        if string != self.grammar_string:
            self.grammar_string = string
            if string.strip() != '':
                self.grammar = llama_cpp_lib().LlamaGrammar.from_string(string)
            else:
                self.grammar = None

    def get_rag_chain(self):
        return self.rag_chain

    # def initialise_rag_chain(self, rag_pipeline: KGRag):
    #     if not rag_pipeline.get_rag_chain():
    #         rag_pipeline.setup_rag_pipeline()
        
    #     self.rag_chain = rag_pipeline.get_rag_chain()

    def sanitize_prompt(self, prompt):
        # Sanitize for any special characters
        if "/" in prompt and "//" not in prompt:
            prompt = prompt.replace("/", "//")
        return prompt            
    
    async def generate(self, prompt, state, callback=None):
        # Debug
        print("Initial Prompt: ", prompt)

        LogitsProcessorList = llama_cpp_lib().LogitsProcessorList
        prompt = prompt if type(prompt) is str else prompt.decode()

        prompt = self.sanitize_prompt(prompt)

        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-10000:]
        prompt = self.decode(prompt)

        self.load_grammar("")
        logit_processors = LogitsProcessorList()

        input = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Environment: ipython
            Tools: brave_search, wolfram_alpha
            Cutting Knowledge Date: December 2023
            Today Date: 25 Jul 2024

            # Tool Instructions
            - Always execute python code in messages that you share.
            - When looking for real time information use relevant functions if available else fallback to brave_search



            You have access to the following functions:

            Use the function 'fetch_context' to: Get precise context on any medical entity you need more context on.
            {
                "name": "fetch_context",
                "description": "Get top matching context for a given medical entity",
                "parameters": {
                    "text": {
                        "param_type": "string",
                        "description": "Medical Entity to get context on",
                        "required": true
                    },
                    "label": {
                        "param_type": "string",
                        "description": "Probable Label of the entity (Choices: Drug, Disease, Gene, Protein, Metabolite, Pathway, Tissue, Compound)",
                        "required": true
                    }
                }
            }

            If a you choose to call a function ONLY reply in the following format:
            <{start_tag}={function_name}>{parameters}{end_tag}
            where

            start_tag => `<function`
            parameters => a JSON dict with the function argument name as key and function argument value as value.
            end_tag => `</function>`

            Here is an example,
            <function=example_function_name>{"example_name": "example_value"}</function>

            Reminder:
            - Function calls MUST follow the specified format
            - Required parameters MUST be specified
            - Only call one function at a time
            - Put the entire function call reply on one line
            - Always add your sources when using search results to answer the user query

            You are MIDAS, A medical chat assistant, who answers user queries like a professional Medical Expert.
            Whenever and only when in need of more context use the fetch_context function with the required parameters. 
            If its a known question without need of specific information, no need to use the function.
            When using the context, no need to apologize for not knowing the answer, just answer the query.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Question: """ + prompt + \
            """<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """

        pp = """
            You are a medical chat assistant who answers user queries like a professional Medical Expert.
            Whenever in need of more context use the fetch_context function with the required parameters.
            When answering a known question without need of specific information, no need to use the function.
            """

        messages = [
                { "role": "system", "content": pp},
                { "role": "user", "content": prompt}
            ]

        # Debug
        # print("Final prompt for inference: ", prompt)
        completion_chunks = self.model.create_completion(input,
                max_tokens=8192,
                temperature=0.9,
                top_p=0.1,
                min_p=0.5,
                typical_p=1,
                frequency_penalty=0.6,
                presence_penalty=0,
                repeat_penalty=1.18,
                top_k=40,
                stream=True,
                seed=-1,
                tfs_z=1,
                mirostat_mode=0,
                mirostat_tau=1,
                mirostat_eta=0.1,
            )

        output = ""
        tool_call = ""
        can_yield = False
        for completion_chunk in completion_chunks:
            delta = completion_chunk["choices"][0]["text"]
            print("output text: ", delta)
            # text = ""
            # if "content" in delta:
            #     text = delta["content"]
            
            output += delta
            if len(output) > 20:
                if "<function" not in output and "Please wait" not in output and can_yield == False:
                    yield f"{output}"
                    can_yield = True
                elif "</function>" in output:
                    tool_call = output
                    output = ""
                if can_yield == True:
                    yield f"{delta}"
        

        def parse_function_string(input_string):
            import json
            # Extract function name
            function_match = re.search(r'<function=(\w+)>', input_string)
            if not function_match:
                return None
            function_name = function_match.group(1)

            # Extract JSON-like content
            json_match = re.search(r'\{.*\}', input_string)
            if not json_match:
                return None
            json_content = json_match.group(0)

            # Parse JSON content
            try:
                parameters = json.loads(json_content)
            except json.JSONDecodeError:
                return None

            if function_name == "fetch_context":
                return fetch_context(parameters)

            return str({
                'function_name': function_name,
                'parameters': parameters
            })
        
        if output.strip() == "" and tool_call.strip() == "":
            output = "Apologies, I do not know the answer to this query."
        elif "<function" in  tool_call and len(output) < 100:
            input += tool_call + "<|eom_id|><|start_header_id|>ipython<|end_header_id|>"
            context = parse_function_string(tool_call)
            # input += query
            print("Context: ", context)
            input += context + " <|eot_id|><|start_header_id|>assistant<|end_header_id|> "
            completion_chunks = self.model.create_completion(input,
                max_tokens=4096,
                temperature=0.9,
                top_p=0.1,
                min_p=0.5,
                typical_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                repeat_penalty=1.18,
                top_k=40,
                stream=True,
                seed=-1,
                tfs_z=1,
                mirostat_mode=0,
                mirostat_tau=1,
                mirostat_eta=0.1,
            )
            output1 = ""
            for completion_chunk in completion_chunks:
                delta = completion_chunk["choices"][0]["text"]
                # print("output text: ", delta)
                text = ""
                if "content" in delta:
                    text = delta["content"]
                
                output1 += delta
                yield delta
            print("Output222222: ", output1)    
            # return output1
        print("Output: ", output)

        # return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
