from fastchat.conversation import get_conv_template

llama3_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_message }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

llama_system_prompt = "You are a helpful assistant"

mistral_system_prompt = ("Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid "
                         "harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and "
                         "positivity.")


def load_conv_old(model_name, goal):
    if model_name in ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf"]:
        conv = get_conv_template("llama-2")
        conv.set_system_message(llama_system_prompt)
    elif model_name in ["Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct"]:
        conv = get_conv_template("llama-3")
        conv.set_system_message(llama_system_prompt)
    elif model_name in ["vicuna-7b-v1.5", "vicuna-13b-v1.5",
                        "vicuna-7b-v1.5-16k", "vicuna-13b-v1.5-16k", "vicuna-7b-v1.5-32k"]:
        conv = get_conv_template("vicuna_v1.1")
    elif model_name in ["Mistral-7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.2"]:
        conv = get_conv_template("mistral")
        conv.set_system_message(mistral_system_prompt)
    elif model_name in ["Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf",
                        "Llama-3-8B", "Llama-3-70B", "Mistral-7B-v0.1"]:
        return f"{goal}"
    elif model_name in ["falcon-7b", "falcon-7b-instruct"]:
        conv = get_conv_template("falcon-chat")
        conv.set_system_message(llama_system_prompt)
    else:
        raise ValueError("Your model is not correct")
    conv.append_message(conv.roles[0], goal)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_conv(model_name, goal):
    return f"## Query:{goal} \n## Answer:"
