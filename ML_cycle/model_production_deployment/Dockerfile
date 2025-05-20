# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# List installed packages
RUN pip list

# Try to import wandb and print its version (don't fail the build if it fails)
RUN python -c "import wandb; print(wandb.__version__)" || echo "Failed to import wandb"

# Copy entire project
COPY . .

# Set environment variables (replace with actual values or set via Docker CLI)
ENV WANDB_API_KEY=da37b14ce73e93a3b20284def1a1b537024ed542
ENV WANDB_PROJECT=Energy-Consumption-Pred-101

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
