FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY htmlloginforms/ htmlloginforms/

RUN uv pip install --system --no-cache .

# data/ and models/ are expected to be mounted at runtime:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models html-login-forms
VOLUME ["/app/data", "/app/models"]

CMD ["train-all"]
