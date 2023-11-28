FROM pytorch/pytorch:latest

WORKDIR /app/inference


COPY . /app/inference
ENV DOCKER_RUN=1

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "inference.py"]
