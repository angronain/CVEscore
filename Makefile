# Docker (recommended on the GPU machine)
dirs:
	mkdir -p data/raw data/processed results checkpoints

build: dirs
	docker compose build

pipeline: dirs          ## data -> baselines + training -> evaluation
	docker compose up --abort-on-container-failure evaluate

pipeline-llm: dirs      ## same, plus the zero-shot LLM baseline (LM Studio running on the host)
	docker compose --profile llm up --abort-on-container-failure evaluate

smoke: dirs             ## 1-minute GPU check on 2,000 CVEs (run after `make data`)
	docker compose run --rm smoke

data: dirs
	docker compose run --rm data

demo:
	docker compose --profile demo up demo

push:
	docker compose run --rm push

# Without Docker (inside a venv)
local-install:
	pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
	pip install -r requirements.txt

local-pipeline: dirs
	python src/fetch_nvd.py
	python src/prepare_data.py
	python src/baselines.py
	python src/train.py
	python src/evaluate.py

.PHONY: dirs build pipeline pipeline-llm smoke data demo push local-install local-pipeline
