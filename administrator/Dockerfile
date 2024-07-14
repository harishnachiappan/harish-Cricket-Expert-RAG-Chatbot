FROM python:3.9
EXPOSE 8083
WORKDIR /app
COPY administrator/requirements.txt ./
RUN pip install -r requirements.txt
COPY administrator/ ./
ENTRYPOINT ["streamlit", "run", "back-end.py", "--server.port=8083", "--server.address=0.0.0.0"]


