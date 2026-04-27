import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

HF_TOKEN = "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"

model = AutoModelForImageTextToText.from_pretrained(
    "stepfun-ai/GOT-OCR-2.0-hf",
    device_map="cpu",
    torch_dtype=torch.float32,
    token=HF_TOKEN
)
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", token=HF_TOKEN)

image = Image.open("sample_images/demo.png")
inputs = processor(image, return_tensors="pt")

generate_ids = model.generate(
    **inputs,
    do_sample=False,
    tokenizer=processor.tokenizer,
    stop_strings="<|im_end|>",
    max_new_tokens=512,
)

result = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)