import streamlit as st
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np
import os
import time
from Upsample import RealESRGAN
# from spaces import GPU

# --- Configuration and Model Loading ---
model_path = "deepseek-ai/Janus-Pro-7B"

@st.cache_resource
def load_models():
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        language_config=language_config,
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    else:
        vl_gpt = vl_gpt.to(torch.float16)

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    sr_model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=2)
    sr_model.load_weights(f'weights/RealESRGAN_x2.pth', download=False)

    return vl_gpt, vl_chat_processor, tokenizer, sr_model

vl_gpt, vl_chat_processor, tokenizer, sr_model = load_models()
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper Functions ---
@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

@torch.inference_mode()
def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    torch.cuda.empty_cache()

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)
    pkv = None

    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])
    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img

@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0):
    torch.cuda.empty_cache()
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    width = 384
    height = 384
    parallel_size = 5

    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=''
        )
        text = text + vl_chat_processor.image_start_tag

        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        stime = time.time()
        ret_images = [image_upsample(Image.fromarray(images[i])) for i in range(parallel_size)]
        print(f'upsample time: {time.time() - stime}')
        return ret_images

# @GPU(duration=60)
def image_upsample(img: Image.Image) -> Image.Image:
    if img is None:
        raise Exception("Image not uploaded")
    width, height = img.size
    if width >= 5000 or height >= 5000:
        raise Exception("The image is too large.")
    global sr_model
    result = sr_model.predict(img.convert('RGB'))
    return result

# --- Streamlit App ---
st.set_page_config(page_title="Janus Pro App", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("Janus Pro App")
    st.markdown(
        "This application demonstrates the capabilities of the Janus Pro model, "
        "including multimodal understanding and text-to-image generation."
    )
    st.markdown("---")
    st.markdown(
        "## Model Parameters"
    )
    default_seed = st.sidebar.number_input("Seed", value=42, min_value=0, step=1)
    st.markdown("---")

    st.markdown(
        "Created by [Ruslan Magana Vsevolodovna](https://ruslanmv.com/)"
    )


# --- Main Content ---
st.title("Multimodal Understanding and Text-to-Image Generation with Janus Pro")

# --- Multimodal Understanding Section ---
with st.expander("**Multimodal Understanding**", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        question_input = st.text_input("Enter your question about the image")
        
    with col2:
        top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
    if image_input and question_input:
        image = Image.open(image_input).convert("RGB")
        image_np = np.array(image)
        with st.spinner("Processing..."):
            response = multimodal_understanding(image_np, question_input, default_seed, top_p, temperature)
        st.text_area("Response", value=response, height=200)
    else:
        st.info("Please upload an image and enter a question to start multimodal understanding.")
    
    # Examples
    st.markdown("**Examples**")
    example_col1, example_col2 = st.columns(2)

    with example_col1:
        if st.button("Example 1: Explain this meme"):
            image_input = "doge.png"  # Replace with your actual example image path
            question_input = "Explain this meme"
            image = Image.open(image_input).convert("RGB")
            image_np = np.array(image)
            with st.spinner("Processing..."):
                response = multimodal_understanding(image_np, question_input, default_seed, top_p, temperature)
            st.image(image, caption="Example Image", use_column_width=True)
            st.text_area("Response", value=response, height=200)

    with example_col2:
        if st.button("Example 2: Convert the formula into latex code."):
            image_input = "equation.png"  # Replace with your actual example image path
            question_input = "Convert the formula into latex code."
            image = Image.open(image_input).convert("RGB")
            image_np = np.array(image)
            with st.spinner("Processing..."):
                response = multimodal_understanding(image_np, question_input, default_seed, top_p, temperature)
            st.image(image, caption="Example Image", use_column_width=True)
            st.text_area("Response", value=response, height=200)

# --- Text-to-Image Generation Section ---
with st.expander("**Text-to-Image Generation**", expanded=True):
    prompt_input = st.text_area("Enter your prompt for image generation", height=150)
    t2i_col1, t2i_col2 = st.columns(2)
    with t2i_col1:
        cfg_weight_input = st.slider("CFG Weight", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    with t2i_col2:
        t2i_temperature = st.slider("T2I Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    
    if st.button("Generate Images"):
        if prompt_input:
            with st.spinner("Generating images..."):
                images = generate_image(prompt_input, default_seed, cfg_weight_input, t2i_temperature)
                st.image(images, width=256)
        else:
            st.warning("Please enter a prompt to generate images.")

    # Examples
    st.markdown("**Examples**")
    
    example_prompts = [
        "Master shifu racoon wearing drip attire as a street gangster.",
        "The face of a beautiful girl",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
        "The image features an intricately designed eye set against a circular backdrop adorned with ornate swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye, there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued with enigmatic allure."
    ]

    for i, example_prompt in enumerate(example_prompts):
        if st.button(f"Example {i+1}"):
            with st.spinner("Generating images..."):
                images = generate_image(example_prompt, default_seed, cfg_weight_input, t2i_temperature)
                st.image(images, width=256)