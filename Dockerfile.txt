FROM python:3.9-slim

# Install system dependencies (Tesseract OCR for image text)
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Hugging Face uses port 7860
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]