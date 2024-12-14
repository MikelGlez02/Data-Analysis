# Dockerfile
# Base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional tools
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . /app/

# Add a script to handle commands
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose application port
EXPOSE 5000

# Command to run the script
CMD ["/app/entrypoint.sh"]
