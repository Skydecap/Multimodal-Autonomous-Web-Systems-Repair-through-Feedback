# MAWSR

MAWSR is a pip-installable library for autonomous web bug analysis and patch review.

## Your exact use case

Install this library in any other web UI project, then run a separate dashboard server (default port `8000`) that is not embedded into your project app.

## Install

```bash
pip install mawsr
```

Or from source:

```bash
pip install .
```

## Run separate dashboard server

```bash
mawsr-dashboard --target-url http://127.0.0.1:3000 --source-dir ./src --port 8000
```

Open:

- `http://127.0.0.1:8000/dashboard`

Notes:

- `--target-url` can point to a locally hosted website or remote URL.
- `--source-dir` can be relative to your current terminal directory or absolute.

## API endpoints

- `GET /dashboard`
- `POST /report`
- `GET /review/analysis`
- `POST /review/apply`
- `GET /review/preview/<filename>`
- `POST /review/push`
- `POST /review/revert`
- `POST /review/feedback`

## Minimal Python usage

```python
from mawsr import WebRepairService

service = WebRepairService(
    target_url="http://127.0.0.1:3000",
    source_dir="./src",
)

result = service.analyze_message("Checkout button does nothing")
print(result["root_cause_analysis"])
```
