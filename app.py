import random
import os

import streamlit as st
import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM


HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)
DEVICE = os.environ.get("DEVICE", "cpu")  # cuda:0
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
MODEL_NAME = os.environ.get("MODEL_NAME", "NbAiLab/nb-gpt-j-6B")
HEADER_INFO = """
# NB-GPT-J-6B
Norwegian GPT-J-6B Model.
""".strip()
SIDEBAR_INFO = """
# Configuration
""".strip()
PROMPT_BOX = "Enter your text..."
EXAMPLES = [
    "Hvem tror dere det er lurest å stemme på til høstens Stortingsvalg?",
    "Hva er verdens beste fotballag?",
    "Vi er en familie på fire med to små barn på 4 og 6. Vi bor i en liten leilighet, men er alle veldig glade i å gå på tur. Nå ønsker vi å skaffe oss hund, men er veldig i tvil om hvilken rase. Er det noen som har noen erfaringer å dele?",
    "Jeg er sikker på at aliens allerede har invadert jorden og lever her iblant oss. Mistenker veldig sterkt av naboen er en alien. Setter stor pris på om noen kan gi noen tips om hvordan jeg kan finne ut av det.",
]


def style():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300&display=swap%22%20rel=%22stylesheet%22" rel="stylesheet">
    <style>
    .ltr,
    textarea {
        font-family: Roboto !important;
        text-align: left;
        direction: ltr !important;
    }
    .ltr-box {
        border-bottom: 1px solid #ddd;
        padding-bottom: 20px;
    }
    .rtl {
        text-align: left;
        direction: ltr !important;
    }
    span.result-text {
        padding: 3px 3px;
        line-height: 32px;
    }
    span.generated-text {
        background-color: rgb(118 200 147 / 13%);
    }
    </style>""", unsafe_allow_html=True)


class Normalizer:
    def remove_repetitions(self, text):
        """Remove repetitions"""
        first_ocurrences = []
        for sentence in text.split("."):
            if sentence not in first_ocurrences:
                first_ocurrences.append(sentence)
        return '.'.join(first_ocurrences)

    def trim_last_sentence(self, text):
        """Trim last sentence if incomplete"""
        return text[:text.rfind(".") + 1]

    def clean_txt(self, text):
        return self.trim_last_sentence(self.remove_repetitions(text))


class TextGeneration:
    def __init__(self):
        self.tokenizer = None
        self.generator = None
        self.task = "text-generation"
        self.model_name_or_path = MODEL_NAME
        set_seed(42)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_auth_token=HF_AUTH_TOKEN if HF_AUTH_TOKEN else None,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, use_auth_token=HF_AUTH_TOKEN if HF_AUTH_TOKEN else None,
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            torch_dtype=DTYPE, low_cpu_mem_usage=False if DEVICE == "cpu" else True
        ).to(device=DEVICE, non_blocking=True)
        _ = self.model.eval()
        self.generator = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)
        # with torch.no_grad():
        # tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
        # gen_tokens = self.model.generate(tokens, do_sample=True, temperature=0.8, max_length=128)
        # generated = tokenizer.batch_decode(gen_tokens)[0]

        # return generated


    def generate(self, prompt, generation_kwargs):
        max_length = len(self.tokenizer(prompt)["input_ids"]) + generation_kwargs["max_length"]
        generation_kwargs["max_length"] = max_length
        # generation_kwargs["num_return_sequences"] = 1
        # generation_kwargs["return_full_text"] = False
        return self.generator(
            prompt,
            **generation_kwargs,
        )[0]["generated_text"]


@st.cache(allow_output_mutation=True)
def load_text_generator():
    generator = TextGeneration()
    generator.load()
    return generator


def main():
    st.set_page_config(
        page_title="NB-GPT-J-6B",
        page_icon="🇳🇴",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    style()
    with st.spinner('Loading the model. Please, wait...'):
        generator = load_text_generator()

    st.sidebar.markdown(SIDEBAR_INFO)

    max_length = st.sidebar.slider(
        label='Max Length',
        help="The maximum length of the sequence to be generated.",
        min_value=1,
        max_value=256,
        value=50,
        step=1
    )
    top_k = st.sidebar.slider(
        label='Top-k',
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering",
        min_value=40,
        max_value=80,
        value=50,
        step=1
    )
    top_p = st.sidebar.slider(
        label='Top-p',
        help="Only the most probable tokens with probabilities that add up to `top_p` or higher are kept for "
             "generation.",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01
    )
    temperature = st.sidebar.slider(
        label='Temperature',
        help="The value used to module the next token probabilities",
        min_value=0.1,
        max_value=10.0,
        value=0.8,
        step=0.05
    )
    do_sample = st.sidebar.selectbox(
        label='Sampling?',
        options=(True, False),
        help="Whether or not to use sampling; use greedy decoding otherwise.",
    )
    do_clean = st.sidebar.selectbox(
        label='Clean text?',
        options=(True, False),
        help="Whether or not to remove repeated words and trim unfinished last sentences.",
    )
    generation_kwargs = {
        "max_length": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "do_clean": do_clean,
    }
    st.markdown(HEADER_INFO)
    prompts = EXAMPLES + ["Custom"]
    prompt = st.selectbox('Examples', prompts, index=len(prompts) - 1)

    if prompt == "Custom":
        prompt_box = PROMPT_BOX
    else:
        prompt_box = prompt

    text = st.text_area("Enter text", prompt_box)
    generation_kwargs_ph = st.empty()
    cleaner = Normalizer()
    if st.button("Generate!"):
        with st.spinner(text="Generating..."):
            generation_kwargs_ph.markdown(", ".join([f"`{k}`: {v}" for k, v in generation_kwargs.items()]))
            if text:
                generated_text = generator.generate(text, generation_kwargs)
                if do_clean:
                    generated_text = cleaner.clean_txt(generated_text)
                if generated_text.strip().startswith(text):
                    generated_text = generated_text.replace(text, "", 1).strip()
                st.markdown(
                    f'<p class="ltr ltr-box">'
                    f'<span class="result-text">{text} <span>'
                    f'<span class="result-text generated-text">{generated_text}</span>'
                    f'</p>',
                    unsafe_allow_html=True
                )

if __name__ == '__main__':
    main()
