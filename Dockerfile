FROM python:3
WORKDIR /app
COPY /src2 /app
RUN pip install -r requirements.txt
