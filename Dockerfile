FROM python:3.11-slim

WORKDIR /code

# Install deps first (separate layer) so Docker can cache this step —
# code changes won't force a full dependency reinstall on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
