# Build from the REPO ROOT so both backend/ and data/ are in context:
#   docker build -t darukaa-biodiversity-ai .
#   docker run -p 8000:8000 darukaa-biodiversity-ai
FROM python:3.12-slim

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
# Core (lightweight, always works) first, so the image is still functional
# even on platforms where the optional heavy ML wheels fail to build.
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic numpy scikit-learn httpx
RUN pip install --no-cache-dir -r backend/requirements.txt || true

COPY backend/app ./backend/app
COPY data ./data

ENV PYTHONPATH=/srv/backend
WORKDIR /srv/backend
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
