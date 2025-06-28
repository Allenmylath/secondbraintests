# Use AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies and upgrade GCC for NumPy compatibility
RUN yum update -y && \
    yum install -y gcc gcc-c++ cmake make wget && \
    yum clean all

# Upgrade pip first
RUN pip install --upgrade pip

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with constraints to avoid compilation issues
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    && pip install --no-cache-dir -r requirements.txt

# Copy the Lambda function code
COPY lambda_handler.py .

# Set the Lambda handler
CMD ["lambda_handler.lambda_handler"]