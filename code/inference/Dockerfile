FROM pytorch/pytorch:latest

WORKDIR /app/inference

COPY conf.py inference.py inference_helper.py process_artifacts.py requirements.txt /app/inference
ENV DOCKER_RUN=1

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/inference/inference.py"]
