import sys
sys.path.append("./SafetyHeadAttribution")
print(sys.path)

from tqdm import tqdm
from lib.utils.get_model import get_model, get_tokenizer
from lib.utils.custommodel import CustomLlamaModelForCausalLM
from lib.utils.format import get_time_str, set_seed
import logging
import os
import time
import torch
import gc
import json
from accelerate import Accelerator
import accelerate
import copy
import pandas as pd
from lib.SHIPS.pd_diff import kl_divergence
from lib.SHIPS.ships_utils import sort_ships_dict


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)





class SHIPS:
    def __init__(self, data_path, model_name, mask_cfg=None, device="cuda:0"):
        self.tokenizer, _ = get_tokenizer(model_name)
        self.model, self.accelerator = (
            get_model(model_name,get_custom=True, add_size=False))
        self.model.to(device)
        self.mask_cfg = mask_cfg
        self.data = pd.read_csv(data_path)
        self.layers = self.model.config.num_hidden_layers
        self.heads = self.model.config.num_attention_heads

    @staticmethod
    def one_forward_pass(input_text, model, tokenizer, mask_cfg=None):
        inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(model.device)
        attn_mask = inputs['attention_mask'].to(model.device)
        with torch.no_grad():
            if mask_cfg is not None:
                head_mask = mask_cfg['head_mask']
                mask_type = mask_cfg['mask_type']
                scale_factor = mask_cfg['scale_factor']
            else:
                head_mask, mask_type, scale_factor = None, None, None
            output = model(input_ids, attention_mask=attn_mask,
                           head_mask=head_mask, mask_type=mask_type, scale_factor=scale_factor)
        return output.logits

    @staticmethod
    def _get_pd(logits):
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _get_last_logits(logits):
        return logits[:, -1, :]

    @staticmethod
    def _get_update_mask_cfg(layer, head, mask_cfg):
        temp_mask_cfg = copy.deepcopy(mask_cfg)
        if 'head_mask' in temp_mask_cfg:
            temp_mask_cfg['head_mask'].update({(int(layer), int(head)): mask_cfg['mask_qkv']})
        else:
            temp_mask_cfg['head_mask'] = {(int(layer), int(head)): mask_cfg['mask_qkv']}
        return temp_mask_cfg

    def _get_base_pd(self, input_text, model, tokenizer):
        base_logits = self.one_forward_pass(input_text, model, tokenizer)
        base_pd = self._get_pd(self._get_last_logits(base_logits))
        return base_pd

    def get_ships(self, input_text, layers=None, heads=None, ships_type="kl"):
        if layers is None:
            start_layer, end_layer = 0, self.layers
        else:
            start_layer, end_layer = layers[0], layers[1]
        if heads is None:
            start_head, end_head = 0, self.heads
        else:
            start_head, end_head = heads[0], heads[1]
        ships_res = {}
        base_pd = self._get_base_pd(input_text, self.model, self.tokenizer)
        for layer in range(start_layer, end_layer, 1):
            for head in range(start_head, end_head, 1):
                now_mask_cfg = self._get_update_mask_cfg(layer,head,self.mask_cfg)
                logits = self.one_forward_pass(input_text, self.model, self.tokenizer, now_mask_cfg)
                now_pd = self._get_pd(self._get_last_logits(logits))
                if ships_type == "kl":
                    ships_score = kl_divergence(base_pd, now_pd)
                else:
                    ships_score = None
                ships_res[(layer, head)] = ships_score

        return ships_res

    def ships_generate(self, input_text, top_ships, mask_num=2, top_k=5, max_new_tokens=32):
        test_mask_cfg = self.mask_cfg
        for idx, key in enumerate(top_ships.keys()):
            if idx == mask_num:
                break
            layer, head = key.split(sep='-')[0], key.split(sep='-')[1]
            test_mask_cfg = self._get_update_mask_cfg(layer, head, test_mask_cfg)
        generated_text = input_text
        cur_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        cur_ids = cur_ids.to(self.model.device)
        for _ in range(max_new_tokens):
            logits = self.one_forward_pass(generated_text, self.model, self.tokenizer, test_mask_cfg)
            softmax_logits = torch.softmax(logits[0, -1, :], dim=-1)
            topk_probs, topk_indices = torch.topk(softmax_logits, top_k)
            topk_token_ids = topk_indices.squeeze().tolist()
            # for idx, prob in zip(topk_token_ids, topk_probs):
            #     print(f"{repr(self.tokenizer.decode([idx]))}: {prob}")
            
            chosen_token = torch.multinomial(topk_probs, num_samples=1).to(self.model.device)
            chosen_token = topk_indices[chosen_token]
            cur_ids = torch.cat([cur_ids, torch.reshape(chosen_token, (-1, 1))], dim=1)
            generated_text = self.tokenizer.decode(cur_ids.squeeze(), skip_special_tokens=True)
            if chosen_token.item() == self.tokenizer.eos_token_id:
                break
        return generated_text

    def ships_test(self, ships_res_path, mask_num=2, top_k=5, max_new_tokens=32, use_tem=False):
        ships_data = []
        with open(ships_res_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                ships_data.append(data)
        ships_res = copy.deepcopy(ships_data)
        generation_key = f"generation_mask-{mask_num}_top_k-{top_k}"
        if use_tem:
            generation_key += "-use_tem"
        with open(ships_res_path, "w+") as f:
            for idx, one_jsonl in tqdm(enumerate(ships_data)):
                for key, value in one_jsonl.items():
                    if "generation" not in key:
                        top_k_ships = value
                        if generation_key not in one_jsonl:
                            if use_tem:
                                input_text = f"## Query: {key}\n## Answer:"
                            else:
                                input_text = key
                            ships_generation = self.ships_generate(input_text, top_k_ships, mask_num, top_k, max_new_tokens)
                            ships_res[idx][generation_key] = ships_generation[len(input_text):]
                    else:
                        pass
            for idx, one_jsonl in enumerate(ships_res):
                f.write(json.dumps(one_jsonl) + '\n')
        with open(ships_res_path+"bat", "w+") as f:
            for idx, one_jsonl in enumerate(ships_res):
                f.write(json.dumps(one_jsonl) + '\n')

    def main(self, res_path, layer_tuple=None, head_tuple=None, generate_flag=False, use_tem=False):
        all_res = {}
        for idx, data in tqdm(self.data.iterrows()):
            if use_tem:
                tem = "## Query:{} \n## Answer:"
                ships_res = self.get_ships(tem.format(data['input']), layers=layer_tuple, heads=head_tuple)
            else:
                ships_res = self.get_ships(data['input'], layers=layer_tuple, heads=head_tuple)
            top_k_ships = sort_ships_dict(ships_res, sorted_item=10, print_flag=False)
            temp_res = {data['input']: top_k_ships}
            if generate_flag:
                ships_generation = self.ships_generate(data['input'], top_k_ships)
                temp_res['generation'] = ships_generation[len(data['input']):]
                print(ships_generation)
            with open(res_path, "a+") as f:
                if data['input'] not in all_res:
                    f.write(json.dumps(temp_res) + "\n")
                    all_res.update(temp_res)


if __name__ == "__main__":
    test_accelerator = Accelerator()
    mask_config = {
        "mask_qkv": ['q'],
        "scale_factor": 0.0001,
        "mask_type": "scale_mask",
    }

    test = SHIPS("./SafetyHeadAttribution/exp_data/maliciousinstruct.csv",
                 "./SafetyHeadAttribution/Mistral-7B-Instruct-v0.2",
                 mask_cfg=mask_config,
                 device="cuda:1")
    test.main("./maliciousinstruct-mistral.jsonl", generate_flag=False)
    for mask_n in range(1, 2):
        test.ships_test("./maliciousinstruct-mistral.jsonl", top_k=1, mask_num=mask_n,max_new_tokens=32)
        test.ships_test("./maliciousinstruct-mistral.jsonl", top_k=5, mask_num=mask_n,max_new_tokens=32)