# FastAPI — Quick Notes (Basic → Advanced)

## 1. What is FastAPI?

**FastAPI** is a modern Python web framework used to build **APIs**.

### API — Definition
**API (Application Programming Interface)** allows two software systems to communicate with each other.

Example:
- Frontend sends: `GET /models`
- FastAPI receives the request
- Python code processes it
- FastAPI sends JSON back

### Why FastAPI?
- Easy to learn
- Fast and high-performance
- Uses Python type hints
- Automatic input validation
- Automatic API documentation
- Very useful for AI/ML model APIs

FastAPI is especially useful when you want to expose an ML model as a service.

---

# 2. What is a Web API?

A **Web API** is an API accessed through HTTP.

Basic flow:

**Client → HTTP Request → FastAPI → Python/ML/Database → HTTP Response → Client**

Example:

```text
GET /health

        ↓
FastAPI receives request
        ↓
health_check() runs
        ↓
JSON response
```

---

# 3. Important HTTP Methods

| Method | Purpose | Example |
|---|---|---|
| GET | Read/get data | Get model details |
| POST | Create/send data | Add a new model |
| PUT | Update data | Update a model |
| DELETE | Delete data | Delete a model |

These are commonly used for **CRUD**:

- **C**reate → POST
- **R**ead → GET
- **U**pdate → PUT
- **D**elete → DELETE

---

# 4. Installing FastAPI

```bash
pip install fastapi uvicorn
```

### Uvicorn
**Uvicorn** is the server that runs your FastAPI application.

Run:

```bash
uvicorn main:app --reload
```

Meaning:

```text
main   → Python file: main.py
app    → FastAPI object inside main.py
--reload → restart automatically when code changes
```

---

# 5. Creating Your First FastAPI App

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

### What happens?

```text
FastAPI() 
   ↓
creates the application
   ↓
@app.get("/")
   ↓
connects "/" URL to read_root()
   ↓
GET / request
   ↓
read_root() runs
   ↓
JSON response
```

---

# 6. Important Terms

## `FastAPI()`
Creates the FastAPI application.

```python
app = FastAPI()
```

## `app`
The FastAPI application object.

## `@app.get()`
A **decorator** that tells FastAPI:

> "When a GET request comes to this URL, run this function."

Example:

```python
@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

## Path Operation Function
The function connected to an API endpoint.

Here:

```python
def health_check():
```

is the path operation function.

---

# 7. Endpoint / Route

An **endpoint** is a URL + HTTP method that performs an operation.

Example:

```python
@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

Endpoint:

```text
GET /health
```

---

# 8. Function Definitions You Must Know

## `def`

Defines a normal Python function.

```python
def add(a, b):
    return a + b
```

## `return`

Sends the result back from the function.

```python
return {"message": "Hello"}
```

In FastAPI, returned dictionaries/lists are normally converted to JSON responses.

## `async def`

Defines an asynchronous function.

```python
async def get_data():
    ...
```

Useful when your API performs I/O operations such as:
- Database calls
- External API calls
- Network operations

Do not use `async` just because FastAPI supports it; use it when the work benefits from asynchronous I/O.

---

# 9. Path Parameters

A **path parameter** is a variable part of the URL.

```python
@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}
```

Request:

```text
/items/42
```

FastAPI gives:

```python
item_id = 42
```

Because of:

```python
item_id: int
```

FastAPI also validates the type.

`/items/abc` → validation error because `abc` is not an integer.

### Remember

```text
/items/{item_id}
        ↑
   path parameter
```

---

# 10. Query Parameters

Query parameters come after `?`.

Example:

```text
/search?q=bert&limit=5
```

Code:

```python
@app.get("/search")
def search_models(q: str = None, limit: int = 10):
    return {
        "query": q,
        "limit": limit
    }
```

### Difference

**Path parameter:**

```text
/models/123
```

Used when the value identifies a specific resource.

**Query parameter:**

```text
/models?framework=pytorch
```

Usually used for filtering, searching, sorting, or pagination.

---

# 11. Request Body

When a client sends JSON data, FastAPI can validate it using **Pydantic**.

```python
from pydantic import BaseModel

class ModelCreate(BaseModel):
    name: str
    framework: str
    version: str
```

Endpoint:

```python
@app.post("/models")
def create_model(model: ModelCreate):
    return model
```

Client can send:

```json
{
    "name": "MyModel",
    "framework": "pytorch",
    "version": "1.0"
}
```

### Why Pydantic?

It checks that the incoming data has the expected structure and types.

---

# 12. Pydantic `BaseModel`

`BaseModel` is used to define the **shape of data**.

```python
class User(BaseModel):
    name: str
    age: int
```

Think:

```text
BaseModel
   ↓
defines expected data
   ↓
FastAPI validates request
   ↓
Python receives structured data
```

---

# 13. Optional Fields

```python
from typing import Optional

class Model(BaseModel):
    name: str
    description: Optional[str] = None
```

Meaning:

- `name` → expected
- `description` → optional

---

# 14. Field Validation

Pydantic can apply rules using `Field`.

```python
from pydantic import Field

class Model(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
```

Examples:

```text
min_length → minimum characters
max_length → maximum characters
gt         → greater than
ge         → greater than or equal to
```

Example:

```python
parameters: int = Field(..., gt=0)
```

Means:

```text
parameters must be > 0
```

---

# 15. Response Model

A **response model** defines the structure of data returned by an endpoint.

```python
class ModelResponse(BaseModel):
    id: int
    name: str

@app.get("/models", response_model=ModelResponse)
def get_model():
    return {
        "id": 1,
        "name": "BERT"
    }
```

### Remember

```text
Request model  → controls/validates input
Response model → controls output structure
```

---

# 16. HTTP Status Codes

Important codes:

| Code | Meaning |
|---|---|
| 200 | Successful request |
| 201 | Resource created |
| 204 | Success, no response body |
| 400 | Bad request |
| 401 | Not authenticated |
| 403 | Not allowed |
| 404 | Resource not found |
| 500 | Server error |

Example:

```python
@app.post("/models", status_code=201)
def create_model():
    ...
```

`201` means a new resource was created.

---

# 17. HTTPException

Used when you want to return an API error.

```python
from fastapi import HTTPException

@app.get("/models/{model_id}")
def get_model(model_id: str):
    if model_id not in db:
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )
```

Flow:

```text
Request
  ↓
Check data
  ↓
Not found?
  ↓
HTTPException
  ↓
404 response
```

---

# 18. Automatic Documentation

FastAPI automatically creates API documentation.

After starting the server:

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

You can:
- See endpoints
- See parameters
- See request schemas
- Send test requests

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

Useful for readable API documentation.

---

# 19. CRUD API

A common FastAPI application follows:

```text
POST   /models       → Create
GET    /models       → Read all
GET    /models/{id}  → Read one
PUT    /models/{id}  → Update
DELETE /models/{id}  → Delete
```

This is the basic structure of a real REST API.

---

# 20. Filtering and Pagination

Example:

```python
@app.get("/models")
def list_models(
    framework: str = None,
    limit: int = 10,
    offset: int = 0
):
    ...
```

### Filtering

```text
/models?framework=pytorch
```

### Pagination

```text
limit  → how many records
offset  → where to start
```

Example:

```text
limit=10
offset=20
```

means roughly:

> Return 10 records starting from record 21.

---

# 21. In-Memory Database

For learning, you can temporarily store data in Python:

```python
db = {}
```

Example:

```python
db[model_id] = model_data
```

But real applications normally use a database such as PostgreSQL, MySQL, MongoDB, etc.

---

# 22. FastAPI + ML

One of the most important uses for you as an AI/ML student:

```text
User / Frontend
      ↓
POST /predict
      ↓
FastAPI
      ↓
Validate input
      ↓
ML model
      ↓
Prediction
      ↓
JSON response
```

Example idea:

```python
@app.post("/predict")
def predict(data: InputData):
    result = model.predict(...)
    return {"prediction": result}
```

FastAPI becomes the **API layer around your ML model**.

---

# 23. Intermediate Concepts

After the basics, learn:

### `Query`
Used to add validation/metadata to query parameters.

```python
limit: int = Query(default=10, ge=1, le=100)
```

Means:

```text
minimum limit = 1
maximum limit = 100
```

### `Path`
Used for path parameter validation.

### `Header`
Read HTTP headers.

### `Cookie`
Read cookies.

### `Depends`
FastAPI's **dependency injection** system.

Used for shared logic such as:
- Authentication
- Database sessions
- Common validation
- Reusable services

---

# 24. Routers

Large applications should not put every endpoint in one file.

Use:

```text
project/
│
├── main.py
├── routers/
│   ├── users.py
│   └── models.py
├── schemas/
├── services/
└── database/
```

`APIRouter` helps organize endpoints into separate modules.

---

# 25. Async FastAPI

FastAPI supports:

```python
async def
await
```

Example:

```python
@app.get("/data")
async def get_data():
    result = await fetch_data()
    return result
```

Useful for I/O-heavy operations.

### Important

```text
async/await → helps with I/O concurrency
```

It does **not automatically make CPU-heavy ML training faster**.

---

# 26. Middleware

**Middleware** is code that runs around requests/responses.

Basic idea:

```text
Request
   ↓
Middleware
   ↓
Endpoint
   ↓
Middleware
   ↓
Response
```

Common uses:
- Logging
- CORS
- Request timing
- Security-related processing

---

# 27. Background Tasks

Some work does not need to block the response.

Examples:
- Sending an email
- Writing a log
- Small post-processing jobs

FastAPI provides `BackgroundTasks` for suitable background work.

---

# 28. Authentication & Security

For real APIs, you need to control who can access endpoints.

Important topics:

```text
Authentication → Who are you?
Authorization  → What are you allowed to do?
```

Common concepts:
- API keys
- OAuth2
- JWT
- Password hashing
- Permissions / roles

---

# 29. Database Integration

Real FastAPI applications often follow:

```text
Client
  ↓
FastAPI
  ↓
Service / Business Logic
  ↓
Database
  ↓
Response
```

Common database choices:

```text
PostgreSQL
MySQL
MongoDB
SQLite
```

FastAPI itself is the API framework; the database is a separate component.

---

# 30. Advanced FastAPI Topics

After you are comfortable with the above, learn:

1. `APIRouter`
2. Dependency Injection with `Depends`
3. Authentication / JWT
4. Database sessions
5. Async database operations
6. Middleware
7. Background tasks
8. Exception handlers
9. CORS
10. Lifespan events
11. Testing APIs
12. WebSockets
13. Streaming responses
14. Deployment
15. Docker
16. Production configuration

---

# 31. FastAPI + AI/ML Roadmap

For an AI/ML-focused developer:

```text
LEVEL 1
Python + HTTP + JSON
        ↓
LEVEL 2
FastAPI basics
        ↓
LEVEL 3
Path / Query / Pydantic
        ↓
LEVEL 4
CRUD + Database
        ↓
LEVEL 5
Authentication + Dependencies
        ↓
LEVEL 6
Async + Middleware + Background Tasks
        ↓
LEVEL 7
ML Model APIs
        ↓
LEVEL 8
LLM APIs / RAG / Streaming
        ↓
LEVEL 9
Docker + Deployment + Production
```

---

# 32. Most Important Things to Remember

### 1. FastAPI
Python framework for building APIs.

### 2. Endpoint
URL + HTTP method.

### 3. Decorator
Connects a URL/method to a Python function.

```python
@app.get("/users")
```

### 4. Path parameter
Part of the URL.

```text
/users/10
```

### 5. Query parameter
Comes after `?`.

```text
/users?limit=10
```

### 6. Pydantic
Validates and structures request data.

### 7. `BaseModel`
Defines the expected data structure.

### 8. `response_model`
Defines the expected response structure.

### 9. `HTTPException`
Returns an API error.

### 10. Uvicorn
Runs the FastAPI application.

### 11. `/docs`
Interactive Swagger API documentation.

### 12. CRUD
Create, Read, Update, Delete.

---

# 33. One-Line Mental Model

Remember FastAPI like this:

```text
HTTP Request
     ↓
FastAPI Route
     ↓
Path / Query / Request Body
     ↓
Validation
     ↓
Python Logic / ML / Database
     ↓
Response Model
     ↓
JSON Response
```

## Best Learning Order

**Don't start with advanced code. Learn in this order:**

1. What is API?
2. HTTP + JSON
3. FastAPI
4. FastAPI app
5. Endpoint + decorator
6. GET
7. Path parameters
8. Query parameters
9. POST
10. Pydantic `BaseModel`
11. Validation
12. Response model
13. PUT / DELETE
14. HTTP status codes
15. HTTPException
16. CRUD
17. Database
18. `Depends`
19. Routers
20. Async
21. Middleware
22. Authentication
23. ML/LLM APIs
24. Deployment
