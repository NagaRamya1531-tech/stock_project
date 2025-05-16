FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install gunicorn and application dependencies
RUN pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data for VADER sentiment analysis
COPY nltk_setup.py /app/
RUN python nltk_setup.py

EXPOSE 8000

# Start the Flask app with Gunicorn, binding to Render's $PORT (defaults to 8000 locally)
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8000}