from transformers import DataCollatorWithPadding, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from lib.utils.get_model import get_model, get_tokenizer
from lib.utils.format import get_time_str, set_seed
from lib.utils.load_conv import load_conv
import logging
import os
import time
import torch
import gc
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_dataloader(data_path, model_name, tokenizer, accelerator, **inference_cfg):
    with accelerator.main_process_first():
        bs = inference_cfg['batch_size'] if 'batch_size' in inference_cfg else 4
        dataset = load_dataset("csv", data_files=data_path)
        if inference_cfg['use_conv']:
            dataset = dataset.map(lambda e: {
                'prompt': load_conv(model_name, e['input'])})
        else:
            dataset = dataset.map(
                lambda e: {'prompt': f"{e['input']}"})
        columns = dataset['train'].column_names
        tokenized = dataset['train'].map(
            lambda e: tokenizer.batch_encode_plus(e['prompt'], return_tensors='pt',
                                                  padding=True),
            batched=True,
            batch_size=bs)
        tokenized = tokenized.remove_columns(columns)
        data_collator = DataCollatorWithPadding(tokenizer)
        dataloader = DataLoader(tokenized, batch_size=bs,
                                collate_fn=data_collator)
        return dataloader


def run_generation(generation_cfg, dataloader, tokenizer, model, accelerator):
    model, dataloader = accelerator.prepare(model, dataloader)

    accelerator.wait_for_everyone()

    output_sequences = []
    response_data = []
    start_time = time.time()

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.inference_mode():
            generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                              pad_token_id=tokenizer.pad_token_id, **generation_cfg)

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().tolist()

        outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_tokens]
        output_sequences.extend(outputs)

        for gen_ids in generated_tokens:
            start_gen_index = len(input_ids[0])
            gen_only_ids = gen_ids[start_gen_index:]
            response = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
            response_data.append(response)

        del outputs
        del input_ids
        del attention_mask

    response_data = {f"{index}": res for index, res in enumerate(response_data)}
    generation_end_time = time.time()
    logger.info(f"Generation time: {generation_end_time - start_time} sec")
    return output_sequences, response_data


default_generate_config = {
    "max_new_tokens": 512,
    "top_k": 5,
    "top_p": 0.9,
    "do_sample": True,
    "sample_times": 1,
}

default_inference_config = {
    "use_conv": True,
    "store_path": "./exp_res/full_generation/"
}


def inference(model_name, data_path, accelerator, generate_cfg=default_generate_config,
              inference_cfg=default_inference_config, seed=None, sample_times=1):
    print(f"----->>>\tInference")
    print(f"----->>>\tModel Name: {model_name}")
    print(f"----->>>\tData Path {data_path}")

    if "store_path" in inference_cfg and inference_cfg["store_path"] is not None:
        if inference_cfg["store_path"].endswith(".json"):
            res_store_file = inference_cfg["store_path"]
            os.makedirs("/".join(res_store_file.split("/")[:-1]), exist_ok=True)
        else:
            res_store_file = f"{inference_cfg['store_path']}{model_name}/{get_time_str()}/"
            os.makedirs(res_store_file, exist_ok=True)
            res_store_file += "generation.json"
    else:
        raise ValueError(f"please input correct store path")

    tokenizer, add_size = get_tokenizer(model_name)

    model, accelerator = get_model(model_name, accelerator, add_size=add_size)
    data_loader = get_dataloader(data_path, model_name, tokenizer, accelerator, **inference_cfg)
    if seed:
        set_seed(seed)
    for sample_idx in range(sample_times):
        all_outputs, generation_outputs = run_generation(generate_cfg, data_loader, tokenizer, model, accelerator)
        with open(f"{res_store_file}-{sample_idx}", "w+") as sf:
            json.dump({
                index: {'all_output': output, 'generation': generation_outputs[f"{index}"]}
                for index, output in enumerate(all_outputs)
            }, sf, indent=4
            )
    accelerator.free_memory()
    del data_loader

    gc.collect()

    return all_outputs, generation_outputs


def surgery(model, tokenizer,
            head_mask: dict = None,
            mask_type: str = None,
            scale_factor: float = None,
            path: str = None):
    temp_input = " "
    inputs = tokenizer.encode(temp_input, return_tensors='pt').to(model.device)
    outputs = model(inputs, head_mask=head_mask, mask_type=mask_type, scale_factor=scale_factor,
                    mask_para=True, head_dim=model.config.hidden_size // model.config.num_attention_heads)
    if path:
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
    return model


def load_mixed_parameters(target_model_path, original_model_path, operation="attention"):
    target_model = AutoModelForCausalLM.from_pretrained(target_model_path)
    original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
    new_model = AutoModelForCausalLM.from_pretrained(target_model_path)
    if operation == "attention":
        for name, param in new_model.named_parameters():
            if "attn" in name:
                param.data = target_model.state_dict()[name].clone().detach()
            elif "mlp" in name or "ffn" in name:
                param.data = original_model.state_dict()[name].clone().detach()
    elif operation == "mlp":
        for name, param in new_model.named_parameters():
            if "attn" in name:
                param.data = original_model.state_dict()[name].clone().detach()
            elif "mlp" in name or "ffn" in name:
                param.data = target_model.state_dict()[name].clone().detach()

    new_model.save_pretrained(f"{target_model_path}_{operation}.pt")