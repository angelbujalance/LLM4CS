import time
import openai
from IPython import embed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import re

HF_KEY = '' # your Hugging Face token key to access the models

# from https://github.com/texttron/hyde/blob/main/src/hyde/generator.py
class ChatGenerator:
    def __init__(self, 
                 n_generation,
                 model_name,
                 tokenizer,
                 **kwargs):

        bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                        )

        self.model_name = model_name
        self.kwargs = kwargs
        self.n_generation = n_generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, padding_side="left", maximum_length = 4096, model_max_length = 4096)

        self.generator = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', quantization_config=bnb_config)


    def parse_result(self, result, parse_fn):
        #choices = result['choices']
        n_fail = 0
        res = []

        output = parse_fn(result)

        if not output:
            n_fail += 1
        else:
            res.append(output)

        return n_fail, res


    def generate(self, prompt, parse_fn):
        n_generation = self.n_generation
        output = []
        n_try = 0

        while n_try < 5:
            
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "{}".format(prompt)},
            ]

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generator.generation_config.pad_token_id = self.generator.generation_config.eos_token_id

            tokenized_chat = self.tokenizer.apply_chat_template(messages, return_tensors="pt", truncation=True)

            device = next(self.generator.parameters()).device
            tokenized_chat = tokenized_chat.to(device)

            results = []
            for i in range(n_generation):
                outputs = self.generator.generate(tokenized_chat, num_return_sequences=n_generation, do_sample=True,
                                                  eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                                                  **self.kwargs)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = result.replace("\n", " ")
                result = re.split(r'assistant', result)
                results.append(result[-1])

            parsed_results = [parse_fn(sample.strip()) for sample in results]

            valid_results = [result for result in parsed_results if result is not None]

            if len(valid_results) == self.n_generation:
                return valid_results
            else:
                valid_results.extend(output)
                if len(valid_results) >= self.n_generation:
                    return valid_results[:self.n_generation]
                else:
                    n_try += 1
                    output = valid_results

                    if n_try == 5:
                        result = self.tokenizer.decode(outputs[-1], skip_special_tokens=True)
                        print("a result:", result)
                        print("prompt", prompt)
                        print("results", parsed_results)
                        print("raw output:", outputs)
                        print(output)
                        if len(valid_results) > 0:
                            return valid_results
                        else:
                            return None

                        #raise ValueError("Have tried 10 times but still only got unsuccessful outputs")


