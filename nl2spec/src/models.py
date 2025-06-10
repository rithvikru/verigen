import os
from statistics import mode

import openai
import requests
import vertexai
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from vertexai.preview.language_models import TextGenerationModel, CodeGenerationModel


import prompting


def _load_key(args, env_var, filename):
    """Load API key from args, environment variable, or keydir."""
    if getattr(args, "keyfile", ""):
        keyfile = args.keyfile
        if os.path.exists(keyfile):
            return open(keyfile).readline().strip("\n")
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value.strip()
    if getattr(args, "keydir", ""):
        keyfile = os.path.join(args.keydir, filename)
        if os.path.exists(keyfile):
            return open(keyfile).readline().strip("\n")
    raise Exception("No key provided.")


def gpt_35_turbo(args):
    key = _load_key(args, "OPENAI_API_KEY", "oai_key.txt")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompting.prompt(args)}],
        n=n,
        temperature=args.temperature,
        stop="FINISH",
    )
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["message"]["content"]
        print("OUTPUT")
        print(output)
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def gpt_4(args):
    key = _load_key(args, "OPENAI_API_KEY", "oai_key.txt")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompting.prompt(args)}],
        n=n,
        temperature=args.temperature,
        stop="FINISH",
    )
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["message"]["content"]
        print("OUTPUT")
        print(output)
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def code_davinci_002(args):
    key = _load_key(args, "OPENAI_API_KEY", "oai_key.txt")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    temperature = args.temperature
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompting.prompt(args),
        temperature=temperature,
        n=n,
        max_tokens=300,
        stop=["FINISH"],
        logprobs=5,
    )
    # print(response["choices"][0]["text"])
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["text"]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def text_davinci_003(args):
    key = _load_key(args, "OPENAI_API_KEY", "oai_key.txt")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    temperature = args.temperature
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompting.prompt(args),
        temperature=temperature,
        n=n,
        max_tokens=300,
        stop=["FINISH"],
        logprobs=5,
    )
    # print(response["choices"][0]["text"])
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["text"]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def code_davinci_edit_001(args):
    key = _load_key(args, "OPENAI_API_KEY", "oai_key.txt")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5

    temperature = args.temperature
    prompt = prompting.prompt(args) + " REPLACE"

    response = openai.Edit.create(
        model="code-davinci-edit-001",
        input=prompt,
        instruction="replace REPLACE with the explanation, an explanation dictionary and the final translation",
        temperature=temperature,
        top_p=1,
        n=n,
    )
    # print(response["choices"][0]["text"])
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["text"][len(prompt) - 8 :].split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def text_bison_001(args):
    key = _load_key(args, "GOOGLE_PROJECT_ID", "google_project_id.txt")
    vertexai.init(project=key)
    model = TextGenerationModel.from_pretrained("text-bison@001")
    n = args.num_tries

    def query():
        return model.predict(
            prompting.prompt(args), temperature=args.temperature, max_output_tokens=300
        )

    choices = []
    for i in range(0, n):
        repsonse = query()
        output = repsonse.text.split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def code_bison_001(args):
    key = _load_key(args, "GOOGLE_PROJECT_ID", "google_project_id.txt")
    vertexai.init(project=key)
    model = CodeGenerationModel.from_pretrained("code-bison@001")
    n = args.num_tries

    def query():
        return model.predict(
            prefix=prompting.prompt(args),
            temperature=args.temperature,
            max_output_tokens=300,
        )

    choices = []
    for i in range(0, n):
        repsonse = query()
        output = repsonse.text.split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def bloom(args):
    n = args.num_tries
    input_prompt = prompting.prompt(args)
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    key = _load_key(args, "HF_API_TOKEN", "hf_key.txt")
    headers = {"Authorization": "Bearer " + key}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    choices = []
    for i in range(0, n):
        raw_output = query(
            {
                "inputs": input_prompt,
                "options": {"use_cache": False, "wait_for_model": True},
                "parameters": {
                    "return_full_text": False,
                    "do_sample": False,
                    "max_new_tokens": 300,
                    "temperature": args.temperature,
                },
            }
        )
        # shots_count = input_prompt.count("FINISH")
        output = raw_output[0]["generated_text"].split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def bloomz(args):
    n = args.num_tries
    input_prompt = prompting.prompt(args)
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
    key = _load_key(args, "HF_API_TOKEN", "hf_key.txt")
    headers = {"Authorization": "Bearer " + key}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    choices = []
    for i in range(0, n):
        raw_output = query(
            {
                "inputs": input_prompt,
                "options": {"use_cache": False, "wait_for_model": True},
                "parameters": {
                    "return_full_text": False,
                    "do_sample": False,
                    "max_new_tokens": 300,
                    "temperature": args.temperature,
                },
            }
        )
        print("RAW OUTPUT")
        print(raw_output)
        # shots_count = input_prompt.count("FINISH")
        output = raw_output[0]["generated_text"].split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)
