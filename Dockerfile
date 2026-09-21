# The Django voice assistant. Needs a platform that runs a persistent ASGI
# process with websocket support; see docs/DEPLOYMENT.md for why that rules
# out serverless hosts.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    USE_TF=0 \
    TRANSFORMERS_NO_TF=1

WORKDIR /app

# ffmpeg and libsndfile are needed to decode uploaded recordings.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt

# The inference wheels are several gigabytes. Build with
#   --build-arg WITH_ML=1
# to include speech recognition, semantic memory and the screening models.
ARG WITH_ML=0
RUN if [ "$WITH_ML" = "1" ]; then pip install -r requirements-ml.txt; fi

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8000

# Daphne, not gunicorn: the voice assistant is a websocket application.
CMD ["sh", "-c", "python manage.py migrate --noinput && \
     daphne -b 0.0.0.0 -p ${PORT:-8000} --root-path=/app/FinalEclipse/project project.asgi:application"]
