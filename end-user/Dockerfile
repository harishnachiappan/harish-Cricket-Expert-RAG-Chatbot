FROM python:3.9
EXPOSE 8084
WORKDIR /app
COPY end-user/. ./
RUN pip install -r requirements.txt
COPY end-user/ ./
ENTRYPOINT ["streamlit", "run", "cricket-app.py", "--server.port=8084", "--server.address=0.0.0.0"]



