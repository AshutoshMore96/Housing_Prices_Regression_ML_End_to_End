# The slim Python as the base-image
FROM python:3.11-slim


# Setting up the working directory inside the container
WORKDIR /app


# Copy the dependency files first (better caching)
COPY pyproject.toml uv.lock* ./


# Install the uv (dependency manager)
RUN pip install uv
RUN uv sync --frozen --no-dev


# Copy the project files
COPY venv .


# Expose the FastAPI default port for the application
EXPOSE 8000


# Command to run the API with the uvicorn
CMD ["uv", "uvicorn", "src.api.main:app", "--host",
"0.0.0.0", "--port", "8000"]
