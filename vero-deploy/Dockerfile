FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

WORKDIR /app

COPY pyproject.toml .
RUN pip install uv
RUN uv pip install .

CMD ["uv","run","evaluator.py"]
