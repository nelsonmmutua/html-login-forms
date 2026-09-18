FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py features.py train.py train_all.py predict.py retrain.py ./

# data/ and models/ are expected to be mounted at runtime:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models html-login-forms
VOLUME ["/app/data", "/app/models"]

CMD ["python", "train_all.py"]
