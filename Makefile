.PHONY: install install-embeddings run index eval test lint check docker deploy

install:            ## core + dev dependencies
	pip install -r requirements-dev.txt

install-embeddings: ## semantic search and reranking models (PyTorch, larger download)
	pip install -r requirements-embeddings.txt

run:                ## start the API and UI on http://localhost:8000
	uvicorn app.api:app --reload

index:              ## rebuild the search index from data/docs
	python -m scripts.build_index --force

eval:               ## retrieval and refusal metrics, with threshold sweep
	python -m scripts.evaluate --sweep

test:
	pytest -q

lint:
	ruff check .

check: lint test    ## what CI runs

docker:
	docker compose up --build

deploy:             ## push to a Hugging Face Space (needs HF_TOKEN)
	python scripts/deploy_hf_space.py
