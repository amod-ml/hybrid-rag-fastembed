# Chatbot API

This project is a FastAPI-based chatbot API with file ingestion capabilities.

## Endpoints

### 1. Chat Endpoint

- **URL**: `/chat`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "conversation_id": "string",
    "query": "string"
  }
  ```
- **Response**:
  ```json
  {
    "message": "string"
  }
  ```

### 2. File Upload Endpoint

- **URL**: `/upload`
- **Method**: POST
- **Request Body**: Form data with file
- **Response**:
  ```json
  {
    "filename": "string",
    "chunks_inserted": "integer",
    "message": "string"
  }
  ```

### 3. Status Endpoint

- **URL**: `/status`
- **Method**: GET
- **Response**:
  ```json
  {
    "status": "alive"
  }
  ```

## Dockerization

1. Ensure you have Docker installed on your system.
2. Navigate to the project root directory containing the Dockerfile.
3. Build the Docker image:
   ```
   docker build -t chatbot-api .
   ```

## Running the Container Locally

Run the container with the following command:

```
docker run -p 8000:8000 chatbot-api
```

The API will be accessible at `http://localhost:8000`.

## Deploying to EC2

1. Launch an EC2 instance with Amazon Linux 2 or Ubuntu.
2. Install Docker on the EC2 instance:
   ```
   sudo yum update -y
   sudo amazon-linux-extras install docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```
3. Log out and log back in to apply the group changes.
4. Clone your repository or copy the project files to the EC2 instance.
5. Navigate to the project directory and build the Docker image:
   ```
   docker build -t chatbot-api .
   ```
6. Run the container:
   ```
   docker run -d -p 80:8000 chatbot-api
   ```
7. The API will be accessible at `http://<EC2-Public-IP>`.

Make sure to configure your EC2 security group to allow inbound traffic on port 80.

## Environment Variables

Ensure you set up the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: URL for your Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant (if required)

You can set these in the Dockerfile or pass them when running the container:

```
docker run -d -p 80:8000 \
  -e OPENAI_API_KEY=your_key \
  -e QDRANT_URL=your_url \
  -e QDRANT_API_KEY=your_key \
  chatbot-api
```

## Development

To run the project locally for development:

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Navigate to the project directory.
3. Create a virtual environment: `uv venv`
4. Activate the virtual environment: `source .venv/bin/activate`
5. Install dependencies: `uv sync`
6. Run the server: `uvicorn chatbot.main:app --reload`

The API will be accessible at `http://localhost:8000`.
```