FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501
# Default: the API. docker-compose.yml also starts the Streamlit UI from this image.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
