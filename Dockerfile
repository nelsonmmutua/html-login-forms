FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY htmlloginforms/ htmlloginforms/

# data/ and models/ are expected to be mounted at runtime:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models html-login-forms
VOLUME ["/app/data", "/app/models"]

ENV PYTHONPATH=htmlloginforms

CMD ["python", "htmlloginforms/train_all.py"]
