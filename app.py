import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

st.set_page_config(page_title="GOT-OCR Demo", page_icon="📄")

HF_TOKEN = "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"

@st.cache_resource
def load_model():
    model = AutoModelForImageTextToText.from_pretrained(
        "stepfun-ai/GOT-OCR-2.0-hf",
        device_map="cpu",
        torch_dtype=torch.float32,
        token=HF_TOKEN
    )
    processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", token=HF_TOKEN)
    return model, processor

def process_image(image: Image.Image, model, processor) -> str:
    inputs = processor(image, return_tensors="pt")
    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=512,
    )
    return processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

st.title("GOT-OCR 2.0")

tab1, tab2 = st.tabs(["Demo", "Upload"])

with tab1:
    st.header("Demo")
    if st.button("Run Demo", type="primary"):
        with st.spinner("Loading model..."):
            model, processor = load_model()
        st.info("Model loaded. Processing demo image...")
        demo_image = Image.open("sample_images/demo.png")
        st.image(demo_image, caption="Demo Image", width=400)
        with st.spinner("Processing..."):
            result = process_image(demo_image, model, processor)
        st.subheader("Result:")
        st.markdown(result)

        st.markdown("---")
        st.subheader("Expected Output:")
        st.text("R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM\nName/Phone Ext.: (...)")

with tab2:
    st.header("Upload Image")
    uploaded = st.file_uploader("Upload PNG or JPEG", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=400)
        if st.button("Recognize", type="primary"):
            with st.spinner("Loading model..."):
                model, processor = load_model()
            with st.spinner("Processing (may take a few minutes on CPU)..."):
                result = process_image(img, model, processor)
            st.subheader("Result:")
            st.markdown(result)