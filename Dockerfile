FROM python:3
WORKDIR /app
COPY /src /app
RUN pip install -r requirements.txt
CMD python /app/Server/fl_server.py