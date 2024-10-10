# HealthTag Service Backend

This is the backend service for HealthTag, providing endpoints for categorizing and managing medical questions.

## Supported Endpoints

### 1. Categorize Question

This endpoint categorizes a medical question into one or more categories.

- **Endpoint**: `/categorize`
- **Method**: POST
- **Input Schema**:
  ```json
  {
    "question": "string"
  }
  ```
- **Response Schema**:
  ```json
  {
    "categories": [
      "Cardiovascular",
      "Dermatology",
      "Neurology",
      "Oncology",
      "Pediatrics",
      "Endocrinology",
      "Pulmonology",
      "Other"
    ]
  }
  ```

### 2. Categorize and Save Question

To categorize and save a question, you need to provide a question in the request body. The service will categorize the question and save it to the database.

- **Endpoint**: `/categorize-and-save`
- **Method**: POST
- **Input Schema**:
  ```json
  {
    "question": "string"
  }
  ```
- **Response Schema**:
  ```json
  {
    "message": "string",
    "uuid": "string"
  }
  ```

### 3. Get Questions by Category

This endpoint retrieves questions based on the specified category.

- **Endpoint**: `/questions`
- **Method**: GET
- **Query Parameter**: `category` (one of the supported medical categories)
- **Response Schema**:
  ```json
  [
    {
      "uuid": "string",
      "question": "string",
      "categories": [
        "Cardiovascular",
        "Dermatology",
        "Neurology",
        "Oncology",
        "Pediatrics",
        "Endocrinology",
        "Pulmonology",
        "Other"
      ]
    }
  ]
  ```

## Supported Medical Categories
- Cardiovascular
- Dermatology
- Neurology
- Oncology
- Pediatrics
- Endocrinology
- Pulmonology
- Other

## Running the Application

To run this application, follow these steps:


1. Install `uv`:
   - Visit the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for detailed instructions.
   - You can use the standalone installer:
     ```
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

2. Navigate to the project directory:
   ```
   cd /poc/service-be
   ```

3. Sync dependencies:
   ```
   uv sync
   ```

4. Run the FastAPI development server:
   ```
   uv run fastapi dev main.py
   ```

The server should now be running and accessible at `http://localhost:8000`.

This is a POC and is not intended for production use.

This service is designed to be run in a local environment. If you want to run this service in a production environment, you need to configure the `OPENAI_API_KEY` and `MONGODB_URI` environment variables.
