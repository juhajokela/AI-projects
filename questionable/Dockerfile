# setup
FROM python:3.11.5

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app

# run
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["streamlit"]
CMD ["run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats false"]
