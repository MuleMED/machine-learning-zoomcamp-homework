FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
