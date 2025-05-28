import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
# from transformers import T5EncoderModel
from peft import PeftModel
import datetime
import json
import os
import sys
import argparse
# import oxen
import time
import pandas as pd

MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"

def get_default_pipeline():
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe

def get_lora_pipeline(path):
    transformer = Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.float16)

    this_dir = os.path.dirname(__file__)
    perf_ckpt = os.path.join(this_dir, f"{path}")

    transformer = PeftModel.from_pretrained(transformer, perf_ckpt)
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID,
                                               transformer=transformer,
                                               # text_encoder=text_encoder,
                                               torch_dtype=torch.float16)
    pipe.to("cuda")

    return pipe

def generate_image(pipe, prompt, i, current_time, prefix, output_dir):
    image = pipe(prompt, num_inference_steps=20).images[0]

    image_base_dir = "generated"
    full_image_dir = os.path.join(output_dir, image_base_dir)

    if not os.path.exists(full_image_dir):
        os.mkdir(full_image_dir)

    file_name = os.path.join(image_base_dir, f"{current_time}_{prefix}_img_{i}.png")
    print(f"Saving image to {file_name}")

    info_json = {
        "prompt": prompt,
        "file_name": f"{file_name}",
    }

    image.save(os.path.join(output_dir, file_name))
    return info_json

def create_prompts():
    # read test_prompts.jsonl
    prompts = []
    with open("test_prompts.jsonl", "r") as jsonl_file:
        for line in jsonl_file:
            prompts.append(json.loads(line)["prompt"])
    return prompts

def generate_images(default_pipe, model_name, trigger_prompt, output_dir, n=-1):
    prompts = create_prompts()
    if n > 0:
        prompts = prompts[:n]

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    results = []
    for i, prompt in enumerate(prompts):
        # time the generation
        base_start_time = time.time()
        json_base = generate_image(default_pipe, prompt, i, current_time, model_name, output_dir)
        base_end_time = time.time()
        print(f"Time taken for base image generation: {base_end_time - base_start_time} seconds")

        trigger_prompt = f"{prompt} {trigger_prompt}"
        trigger_start_time = time.time()
        json_trigger = generate_image(default_pipe, trigger_prompt, i, current_time, f"{model_name}_w_trigger", output_dir)
        trigger_end_time = time.time()
        print(f"Time taken for trigger image generation: {trigger_end_time - trigger_start_time} seconds")

        avg_time = (base_end_time - base_start_time + trigger_end_time - trigger_start_time) / 2

        combined_json = {
            "id": i,
            "prompt": prompt,
            "trigger_prompt": trigger_prompt,
            "base_image": json_base["file_name"],
            "trigger_image": json_trigger["file_name"],
            "avg_time": avg_time,
        }
        results.append(combined_json)
    return results

if __name__ == "__main__":
    # Use argparse to specify the model and oxen repository
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to the LoRA model")
    # parser.add_argument("-r", "--repo", required=True, help="Path to the Oxen repository")
    # parser.add_argument("-f", "--filename", required=True, help="Path to the Oxen repository")
    # parser.add_argument("-n", "--num_results", type=int, default=-1, help="Number of prompts to process")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt to use for the images")
    parser.add_argument("-o", "--output", required=True, help="The output directory to save the images")
    args = parser.parse_args()

    lora_path = args.model

    default_pipe = get_lora_pipeline(lora_path)
    prompt = args.prompt
    current_time = time.time()
    trigger_prompt = ""
    i = 0
    prefix = ""
    output_path = args.output

    json_trigger = generate_image(default_pipe, prompt, i, current_time, prefix, output_path)
    print(json_trigger)

    # Write a loop that iterates over a parquet file named 'test.parquet' and generates images for each row
    # df = pd.read_parquet(os.path.join(output_path, 'test.parquet'))
    # df['image'] = None
    # for index, row in df.iterrows():
    #     print("Processing row:", index)
    #     prompt = row['prompt']
    #     i = index
    #     current_time = time.time()
    #     prefix = ""
    #     json_trigger = generate_image(default_pipe, prompt, i, current_time, prefix, output_path)
    #     print(json_trigger)
    #     df.loc[index, 'image'] = json_trigger['file_name']

    # df.to_parquet(os.path.join(output_path, 'test.parquet'))



    # output_path = args.repo
    # num_results = args.num_results

    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    # output_file_name = args.filename
    # output_file_path = os.path.join(output_path, output_file_name)

    # base_results = []
    # # just to scope this model or we run out of memory
    # if True:
    #     default_pipe = get_default_pipeline()
    #     base_results = generate_images(default_pipe, "base", args.prompt, output_path, n=num_results)

    #     del default_pipe
    #     torch.cuda.empty_cache()

    # lora_pipe = get_lora_pipeline(lora_path)
    # lora_results = generate_images(lora_pipe, "lora", args.prompt, output_path, n=num_results)

    # results = []
    # for base, lora in zip(base_results, lora_results):
    #     results.append({
    #         "id": base["id"],
    #         "prompt": base["prompt"],
    #         "trigger_prompt": base["trigger_prompt"],
    #         "base_image": base["base_image"],
    #         "trigger_image": base["trigger_image"],
    #         "lora_base_image": lora["base_image"],
    #         "lora_trigger_image": lora["trigger_image"],
    #         "avg_time": base["avg_time"],
    #     })

    # print(results)
    # with open(output_file_path, "w") as jsonl_file:
    #     for result in results:
    #         jsonl_file.write(json.dumps(result))
    #         jsonl_file.write("\n")

    # metadata = json.dumps({
    #     "_oxen": {
    #         "render": {
    #             "func": "image"
    #         }
    #     }
    # })

    # # Add and push to oxen
    # repo = oxen.LocalRepo(output_path)
    # repo.add("results")
    # repo.add(output_file_name)
    # repo.add_schema_metadata(output_file_name, "base_image", metadata)
    # repo.add_schema_metadata(output_file_name, "trigger_image", metadata)
    # repo.add_schema_metadata(output_file_name, "lora_base_image", metadata)
    # repo.add_schema_metadata(output_file_name, "lora_trigger_image", metadata)
    # repo.commit("Adding generated images")
    # repo.push()