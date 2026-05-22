# Final Project Checklist

## Setup

- [done] Fork <https://github.com/ishraq-hassan/MLOps-Course-Labs>
- [done] Clone your fork locally
- [done] Install dependencies: `uv sync` or `pip install -e ".[dev]"`
- [done] Install pre-commit hooks: `uv run pre-commit install`
- [done] Place your best churn model into `data/model.joblib`

## Logger (`app/logger_setup.py`)

- [done] TODO 1: Set up basic logging with level INFO
- [done] TODO 2: Create and return a named logger

## Model Utils (`app/model_utils.py`)

- [done] TODO 1: Load your model (and preprocessor) at module level
- [done] TODO 2: Implement `preprocess()`
- [done] TODO 3: Call `preprocess()` inside `predict_churn()`
- [done] TODO 4: Implement `predict_churn()` using the model
- [ ] TODO 5: Fill in sample features
- [ ] Verify: `uv run python -m app.model_utils`

## API (`main.py`)

- [done] TODO 1: Define `ChurnRequest` fields
- [done] TODO 2: Create `GET /`
- [done] TODO 3: Create `GET /health`
- [done] TODO 4: Create `POST /predict` with logging
- [done] TODO 5: Register handlers in `Litestar(route_handlers=[...])`

## Run & Screenshot

- [done] Start the server: `uv run litestar --app main:app run --reload`
- [done] Open <http://localhost:8000/schema/swagger>
- [done] **Take a screenshot of the Swagger UI**

## Tests (`tests/test_main.py`)

- [done] TODO 1: Function test for `predict_churn`
- [done] TODO 3: Endpoint test for `POST /predict`
- [done] TODO 4: Endpoint test for `GET /health`
- [done] TODO 5: Endpoint test for `GET /`
- [done] Run: `uv run pytest tests/ -v --cov=app --cov=main --cov-report=term-missing`
- [done] **Coverage is above 70%**
- [done] **Take a screenshot of the results + coverage**

## Bonus

- [ ] TODO 2 (tests): Extra function test with edge cases
- [ ] TODO 6 (tests): Test invalid input returns 400
- [ ] Set up HyperDX for live logs

## Submit

- [x] Commit the `uv.lock` file (**points will be deducted if missing**)
- [x] Push to your fork
- [x] **Upload** the Swagger UI screenshot
- [x] **Upload** the test results and coverage screenshot
- [x] **Upload** the link to your repo

---

## Quick Reference

```bash
uv run litestar --app main:app run --reload                              # start server
uv run pytest tests/ -v                                                  # run tests
uv run pytest tests/ -v --cov=app --cov=main --cov-report=term-missing   # with coverage
```

| Litestar vs FastAPI     | FastAPI            | Litestar                      |
| ----------------------- | ------------------ | ----------------------------- |
| Swagger UI              | `/docs`            | `/schema/swagger`             |
| Run command             | `uvicorn main:app` | `litestar --app main:app run` |
| Route decorators        | `@app.get("/")`    | `@get("/")`                   |
| App creation            | `FastAPI()`        | `Litestar(route_handlers=[])` |
| POST default status     | 200                | 201                           |
| Validation error status | 422                | 400                           |
