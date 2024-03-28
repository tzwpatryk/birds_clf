FROM pytorch/pytorch
WORKDIR /app

COPY /app /app/app
COPY /models /app/models
COPY run.py /app/
COPY config.py /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 3000

ENV FLASK_APP=./run.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=3000

CMD flask run

