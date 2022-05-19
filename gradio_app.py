import os
import random
import string

import gradio as gr
import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

DEBUG = os.environ.get("DEBUG", "false")[0] in "ty1"
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)
DEVICE = os.environ.get("DEVICE", "cpu")  # cuda:0
if DEVICE != "cpu" and not torch.cuda.is_available():
    DEVICE = "cpu"
logger.info(f"DEVICE {DEVICE}")
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
MODEL_NAME = os.environ.get("MODEL_NAME", "NbAiLab/nb-gpt-j-6B")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 256))
HEADER_INFO = """
# NB-GPT-J-6B
Norwegian GPT-J-6B Model.
""".strip()
LOGO = "https://s3.amazonaws.com/moonup/production/uploads/1644417861130-5ef3829e518622264685b0cd.webp"
HEADER = f"""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300&display=swap%22%20rel=%22stylesheet%22" rel="stylesheet">
<style>
.ltr,
textarea {{
    font-family: Roboto !important;
    text-align: left;
    direction: ltr !important;
}}
.ltr-box {{
    border-bottom: 1px solid #ddd;
    padding-bottom: 20px;
}}
.rtl {{
    text-align: left;
    direction: ltr !important;
}}
span.result-text {{
    padding: 3px 3px;
    line-height: 32px;
}}
span.generated-text {{
    background-color: rgb(118 200 147 / 13%);
}}
</style>
<div align=center>
<img src="{LOGO}" width=150/>

# NB-GPT-J-6B

NB-GPT-J-6B is a GTP-3-like model Norwegian by the [National Library of Norway AI-Lab](https://ai.nb.no).

This model has been trained with [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax) using TPUs provided by Google through the Tensor Research Cloud program, starting off the [GPT-J-6B model weigths from EleutherAI](https://huggingface.co/EleutherAI/gpt-j-6B), and trained on the [Norwegian Colossal Corpus](https://huggingface.co/datasets/NbAiLab/NCC) and other Internet sources. *This demo runs on {DEVICE.split(':')[0].upper()}*.

</div>
"""

FOOTER = """
<div align=center>

For more information, visit the [model repository](https://huggingface.co/NbAiLab/nb-gpt-j-6B).

<img src="https://visitor-badge.glitch.me/badge?page_id=NbAiLab/nb-gpt-j-6B"/>
<div align=center>
""".strip()

EXAMPLES = [
    "",
    "Hvem tror dere det er lurest å stemme på til høstens Stortingsvalg?",
    "Hva er verdens beste fotballag?",
    "Vi er en familie på fire med to små barn på 4 og 6. Vi bor i en liten leilighet, men er alle veldig glade i å gå på tur. Nå ønsker vi å skaffe oss hund, men er veldig i tvil om hvilken rase. Er det noen som har noen erfaringer å dele?",
    "Jeg er sikker på at aliens allerede har invadert jorden og lever her iblant oss. Mistenker veldig sterkt av naboen er en alien. Setter stor pris på om noen kan gi noen tips om hvordan jeg kan finne ut av det.",
]

AGENT = "NB"
USER = "INTERVJUER"
CONTEXT = """Følgende samtale er et utdrag fra et intervju med {AGENT} holdt i Oslo for Norwegian Norsk rikskringkasting AS:

{USER}: Velkommen, {AGENT}. En glede å ha deg hos oss i dag.
{AGENT}: Takk. Gleden er min."""

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
        logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_auth_token=HF_AUTH_TOKEN if HF_AUTH_TOKEN else None,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, use_auth_token=HF_AUTH_TOKEN if HF_AUTH_TOKEN else None,
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
            torch_dtype=DTYPE, low_cpu_mem_usage=False if DEVICE == "cpu" else True
        ).to(device=DEVICE, non_blocking=False)
        _ = self.model.eval()
        device_number = -1 if DEVICE == "cpu" else int(DEVICE.split(":")[-1])
        self.generator = pipeline(self.task, model=self.model, tokenizer=self.tokenizer, device=device_number)
        logger.info("Loading model done.")
        # with torch.no_grad():
        # tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
        # gen_tokens = self.model.generate(tokens, do_sample=True, temperature=0.8, max_length=128)
        # generated = tokenizer.batch_decode(gen_tokens)[0]

        # return generated


    def generate(self, text, generation_kwargs):
        max_length = len(self.tokenizer(text)["input_ids"]) + generation_kwargs["max_length"]
        generation_kwargs["max_length"] = min(max_length, self.model.config.n_positions)
        generated_text = None
        if text:
            for _ in range(10):
                generated_text = self.generator(
                    text,
                    **generation_kwargs,
                )[0]["generated_text"]
                if generation_kwargs["do_clean"]:
                    generated_text = cleaner.clean_txt(generated_text)
                if generated_text.strip().startswith(text):
                    generated_text = generated_text.replace(text, "", 1).strip()
                if generated_text:
                    return (
                        text,
                        text + " " + generated_text,
                        [(text, None), (generated_text, AGENT)]
                    )
            if not generated_text:
                return (
                    "",
                    "",
                    [("Etter 10 forsøk ble ingenting generert. Prøv å endre alternativene.", "ERROR")]
                )
        return (
            "",
            "",
            [("Du må skrive noe først.", "ERROR")]
        )
            # return (text + " " + generated_text,
            #     f'<p class="ltr ltr-box">'
            #     f'<span class="result-text">{text} <span>'
            #     f'<span class="result-text generated-text">{generated_text}</span>'
            #     f'</p>'
            # )


#@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
#@st.cache(allow_output_mutation=True)
#@st.cache(allow_output_mutation=True, hash_funcs={TextGeneration: lambda _: None})
def load_text_generator():
    text_generator = TextGeneration()
    text_generator.load()
    return text_generator

cleaner = Normalizer()
generator = load_text_generator()


def complete_with_gpt(text, max_length, top_k, top_p, temperature, do_sample, do_clean):
    generation_kwargs = {
        "max_length": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "do_clean": do_clean,
    }
    return generator.generate(text, generation_kwargs)

def expand_with_gpt(hidden, text, max_length, top_k, top_p, temperature, do_sample, do_clean):
    generation_kwargs = {
        "max_length": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "do_clean": do_clean,
    }
    return generator.generate(hidden or text, generation_kwargs)

def chat_with_gpt(user, agent, context, user_message, history, max_length, top_k, top_p, temperature, do_sample, do_clean):
    # agent = AGENT
    # user = USER
    generation_kwargs = {
        "max_length": 25,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "do_clean": do_clean,
        # "num_return_sequences": 1,
        # "return_full_text": False,
    }
    message = user_message.split(" ", 1)[0].capitalize() + " " + user_message.split(" ", 1)[-1]
    history = history or []
    context = context.format(USER=user or USER, AGENT=agent or AGENT).strip()
    if context[-1] not in ".:":
        context += "."
    context_length = len(context.split())
    history_take = 0
    history_context = "\n".join(f"{user}: {history_message.capitalize()}.\n{agent}: {history_response}." for history_message, history_response in history[-len(history) + history_take:])
    while len(history_context.split()) > generator.model.config.n_positions - (generation_kwargs["max_length"] + context_length):
        history_take += 1
        history_context = "\n".join(f"{user}: {history_message.capitalize()}.\n{agent}: {history_response}." for history_message, history_response in history[-len(history) + history_take:])
        if history_take >= generator.model.config.n_positions:
            break
    context += history_context
    for _ in range(5):
        response = generator.generate(f"{context}\n\n{user}: {message}.\n", generation_kwargs)[1]
        if DEBUG:
            print("\n-----" + response + "-----\n")
        response = response.split("\n")[-1]
        if agent in response and response.split(agent)[-1]:
            response = response.split(agent)[-1]
        if user in response and response.split(user)[-1]:
            response = response.split(user)[-1]
        if response[0] in string.punctuation:
            response = response[1:].strip()
        if response.strip().startswith(f"{user}: {message}"):
            response = response.strip().split(f"{user}: {message}")[-1]
        if response.replace(".", "").strip() and message.replace(".", "").strip() != response.replace(".", "").strip():
            break
    if DEBUG:
        print()
        print("CONTEXT:")
        print(context)
        print()
        print("MESSAGE")
        print(message)
        print()
        print("RESPONSE:")
        print(response)
    if not response.strip():
        response = random.choice(["Jeg vet ikke helt hvordan jeg skal svare på det.", "Jeg er ikke sikker.", "Jeg vil helst ikke svare.", "Ingen anelse.", "Kan vi endre emnet?"])
    history.append((user_message, response))
    return history, history, ""


with gr.Blocks() as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        with gr.Group():
            with gr.Box():
                gr.Markdown("Alternativer")
            max_length = gr.Slider(
                label='Maksimal lengde',
                minimum=1,
                maximum=MAX_LENGTH,
                value=50,
                step=1
            )
            top_k = gr.Slider(
                label='Top-k',
                minimum=40,
                maximum=80,
                value=50,
                step=1
            )
            top_p = gr.Slider(
                label='Top-p',
                minimum=0.0,
                maximum=1.0,
                value=0.95,
                step=0.01
            )
            temperature = gr.Slider(
                label='Temperatur',
                minimum=0.1,
                maximum=10.0,
                value=0.8,
                step=0.05
            )
            do_sample = gr.Checkbox(
                label='Prøvetaking?',
                value = True,
                # options=(True, False),
            )
            do_clean = gr.Checkbox(
                label='Klartekst?',
                value = True,
                # options=(True, False),
            )
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Generer"):
                    textbox = gr.Textbox(label="Tekst", placeholder="Skriv noe (eller velg et eksempel) og trykk 'Generer'...", lines=8)
                    examples = gr.Dropdown(label="Eksempler", choices=EXAMPLES, value=None, type="value")
                    hidden = gr.Textbox(visible=False, show_label=False)
                    with gr.Box():
                        # output = gr.Markdown()
                        output = gr.HighlightedText(label="Utfall", combine_adjacent=True, color_map={AGENT: "green", "ERROR": "red"})
                    with gr.Row():
                        generate_btn = gr.Button("Generer")
                        generate_btn.click(complete_with_gpt, inputs=[textbox, max_length, top_k, top_p, temperature, do_sample, do_clean], outputs=[textbox, hidden, output])
                        expand_btn = gr.Button("Legg til")
                        expand_btn.click(expand_with_gpt, inputs=[hidden, textbox, max_length, top_k, top_p, temperature, do_sample, do_clean], outputs=[textbox, hidden, output])

                        edit_btn = gr.Button("Redigere", variant="secondary")
                        edit_btn.click(lambda x: (x, "", []), inputs=[hidden], outputs=[textbox, hidden, output])
                        clean_btn = gr.Button("Viske ut", variant="secondary")
                        clean_btn.click(lambda: ("", "", [], ""), inputs=[], outputs=[textbox, hidden, output, examples])
                    examples.change(lambda x: x, inputs=[examples], outputs=[textbox])

                with gr.TabItem("Chatter") as tab_chat:
                    tab_chat.select(lambda: 25, inputs=[], outputs=[max_length])
                    context = gr.Textbox(label="Kontekst", value=CONTEXT, lines=5)
                    with gr.Row():
                        agent = gr.Textbox(label="Middel", value=AGENT)
                        user = gr.Textbox(label="Bruker", value=USER)
                    history = gr.Variable(default_value=[])
                    chatbot = gr.Chatbot(color_map=("green", "gray"))
                    with gr.Row():
                        message = gr.Textbox(placeholder="Skriv meldingen din her og trykk 'Send'", show_label=False)
                        chat_btn = gr.Button("Send")
                    chat_btn.click(chat_with_gpt, inputs=[agent, user, context, message, history, max_length, top_k, top_p, temperature, do_sample, do_clean], outputs=[chatbot, history, message])
    gr.Markdown(FOOTER)



demo.launch()
# gr.Interface(complete_with_gpt, inputs=[textbox, max_length, top_k, top_p, temperature, do_sample, do_clean], outputs=[hidden, output]).launch()
