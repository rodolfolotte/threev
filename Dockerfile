FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/src/app

RUN mkdir -p /usr/src/app/artefacts/
# COPY yolov8m-seg.pt ./artefacts

COPY requirements.txt ./
COPY app.py ./
COPY yolov8m-seg.pt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]