FROM python:3.8
WORKDIR /app
COPY /centralized /app
RUN python --version
RUN pip --version
RUN pip install -r requirements.txt
