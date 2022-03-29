FROM lucone83/streamlit-nginx:python3.8

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

USER streamlitapp
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers tokenizers

ENV MODEL_NAME NbAiLab/nb-gpt-j-6B
ENV HF_AUTH_TOKEN ""

RUN git config --global credential.helper store
RUN mkdir /home/streamlitapp/.huggingface
# RUN echo "${HF_AUTH_TOKEN}" > /home/streamlitapp/.huggingface/token

# COPY --chown=streamlitapp requirements.txt /home/streamlitapp/requirements.txt
COPY --chown=streamlitapp app.py /home/streamlitapp/app.py

# RUN pip3 install -r /home/streamlitapp/requirements.txt

CMD ["streamlit", "run", "/home/streamlitapp/app.py"]
