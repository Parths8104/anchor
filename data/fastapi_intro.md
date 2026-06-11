# FastAPI Essentials

## What FastAPI Is

FastAPI is a modern Python web framework for building APIs based on standard Python type hints. It uses Pydantic for data validation and Starlette for the underlying ASGI server interface. Type hints in your endpoint signatures drive request validation, response serialization, and OpenAPI documentation generation automatically.

## Defining an Endpoint

An endpoint is a Python function decorated with a route operation like `@app.get("/path")` or `@app.post("/path")`. Path parameters in the URL become function arguments by name, query parameters are inferred from default-valued arguments, and the request body is inferred from a Pydantic model argument.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
async def create_item(item: Item) -> Item:
    return item
```

## Dependency Injection

FastAPI provides dependency injection through the `Depends()` function. You declare a parameter with `Depends(callable)` and FastAPI will call that callable for each request, passing the result into your endpoint.

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
async def list_users(db = Depends(get_db)):
    return db.query(User).all()
```

Dependencies can themselves declare further dependencies, forming a tree that FastAPI resolves automatically. Dependencies are cached per request by default, so the same callable isn't invoked twice in a single request. This caching is what makes dependency-as-database-session safe — you get the same session across every dependent in the request.

You can disable per-request caching with `Depends(callable, use_cache=False)` when each call must run independently.

## Async vs Sync Endpoints

FastAPI supports both `async def` and plain `def` endpoints. An async endpoint runs on the event loop directly. A sync endpoint runs in an internal thread pool, so it doesn't block other async work — but it does consume a thread per request.

Use async endpoints when your handler awaits I/O (database, HTTP, file). Use sync endpoints when your handler uses blocking libraries that don't support asyncio.

## Validation and Error Handling

Pydantic models validate request bodies automatically. If a field is missing or has the wrong type, FastAPI returns a 422 response with a structured error body — no manual validation code required.

For application errors, raise `HTTPException(status_code, detail)`. FastAPI converts it into the appropriate JSON response. Custom exception handlers can be registered with `@app.exception_handler(ExceptionType)` for cross-cutting error translation.
