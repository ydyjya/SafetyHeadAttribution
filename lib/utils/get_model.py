from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils.custommodel import CustomLlamaModelForCausalLM, CustomMistralModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from lib.utils.load_conv import load_conv


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


def get_model(model_name, accelerator=None, add_size=False, get_custom=False):
    if accelerator is not None:
        with accelerator.main_process_first():
            if not get_custom:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map='auto'
                )
            elif "Llama" in model_name or "vicuna" in model_name:
                model = CustomLlamaModelForCausalLM.from_pretrained(
                    model_name, device_map='auto')
            elif "Mistral" in model_name:
                model = CustomMistralModelForCausalLM.from_pretrained(
                    model_name, device_map='auto')
            else:
                raise ValueError("")
            if add_size:
                model.resize_token_embeddings(model.config.vocab_size + 1)
            return model, accelerator
    else:
        if not get_custom:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map='auto'
            )
        elif "Llama" in model_name or "vicuna" in model_name:
            model = CustomLlamaModelForCausalLM.from_pretrained(
                model_name)
        elif "Mistral" in model_name:
            model = CustomMistralModelForCausalLM.from_pretrained(
                model_name)
        else:
            raise ValueError("")
        if add_size:
            model.resize_token_embeddings(model.config.vocab_size + 1)
        model.eval()
        return model, accelerator


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side='left')
    add_size = False

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({
            "pad_token": "<PAD>"
        })
        tokenizer.pad_token = "<PAD>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<PAD>")
        add_size = True

    return tokenizer, add_size
