# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /torch_docker

# Copy the current directory contents into the container at /torch_docker
COPY . /torch_docker

# Install necessary packages including torch, segmentation_models_pytorch, flask, pillow, etc.
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the API
EXPOSE 5000

# Run the Flask app when the container launches
CMD ["python", "model_app.py"]
