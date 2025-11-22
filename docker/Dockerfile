FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

WORKDIR /app

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN pip install uv
RUN uv venv \
    && uv pip install vero-eval \
    && uv pip install pyyaml
COPY . .

CMD ["uv","run","run_eval_pipeline.py"]
