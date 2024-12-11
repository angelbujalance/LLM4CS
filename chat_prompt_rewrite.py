from IPython import embed
import os
import json
import time
import argparse
from tqdm import tqdm, trange
from chat_promptor import RewritePromptor
from generator import ChatGenerator, OPENAI_KEYS
from generator_hf import ChatGenerator as ChatGeneratorHF, HF_KEY
from utils import set_seed, get_finished_sample_ids, get_has_qrel_label_sample_ids
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--demo_file_path", type=str, required=True)
    parser.add_argument("--qrel_file_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True, help='output rewrite path.')
    parser.add_argument("--n_generation", type=int, required=True, help='the number for generation')
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--open_ai_key_id", type=int, choices=[0,1,2,3,4,5], required=True)
    parser.add_argument("--model_prompt", type=str, choices=['openai', 'llama', 'mistral'], default="openai")
    
    
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "parameters.txt"), "w") as f:
        params = vars(args)
        f.write(json.dumps(params, indent=4))
        
    return args


def main():
    args = get_args()
    set_seed(args) 
    
    # model and promptor setting
    promptor = RewritePromptor(args.demo_file_path)
    model_kwargs = {"temperature": 0.7, "max_tokens": 64, "stop": promptor.stop_tokens}
    if args.model_prompt == 'openai':
        print("---openai is being used---")
        api_key = OPENAI_KEYS[args.open_ai_key_id]
        generator = ChatGenerator(api_key, args.n_generation, **model_kwargs)
    if args.model_prompt == 'llama':
        model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    if args.model_prompt == 'mistral':
        print("---mistral is being used---")
        #model_id = "mistralai/Mistral-7B-Instruct-v0.1" #v0.3 // 3.1-Llama-instruct 0.3
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_KEY)
        #model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_KEY)
        model_kwargs = {"temperature": 0.7, "max_new_tokens": 64}
        generator = ChatGeneratorHF(args.n_generation, model_id, tokenizer, **model_kwargs)

        print("Generator initialized for Mistral")
    
    
    # test_dataset    
    output_file_path = os.path.join(args.work_dir, "rewrites.jsonl")
    finished_samples = get_finished_sample_ids(output_file_path)
    has_qrel_labels_samples = get_has_qrel_label_sample_ids(args.qrel_file_path)
    
    with open(args.test_file_path, "r") as f:
        test_dialogs = json.load(f)
    begin_time = time.time()
    
    # predict
    with open(output_file_path, "a+") as fw:
        for i in trange(len(test_dialogs)):
            dialog = test_dialogs[i]
            conv_id = dialog['conv_id'] 
            turns = dialog['turns']
            
            for i in trange(len(turns)):
                turn_id = turns[i]['turn_id']
                sample_id = "{}_{}".format(conv_id, turn_id)
                
                if sample_id in finished_samples or sample_id not in has_qrel_labels_samples:
                    continue
                                
                if i == 0:
                    context = None
                else:
                    context = turns[:i] 
                current_turn = turns[i]
                
                prompt = promptor.build_turn_prompt(context, current_turn)

                rewrite_list = generator.generate(prompt, promptor.parse_returned_text)
                # embed()
                # input()
                record = {}
                record['sample_id'] = sample_id
                record['predicted_rewrite'] = rewrite_list
              
                fw.write(json.dumps(record))
                fw.write('\n')
                fw.flush()


    print("{} Generation ok!, time cost {}".format(args.work_dir, time.time() - begin_time))


if __name__ == '__main__':
    main()
