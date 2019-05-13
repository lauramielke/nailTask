FROM python:3.6
WORKDIR /app

COPY requirements.txt /app
COPY model.json /app
COPY model.h5 /app
RUN pip install -r ./requirements.txt

COPY nailTask.py /app
CMD ["python", "nailTask.py"]~