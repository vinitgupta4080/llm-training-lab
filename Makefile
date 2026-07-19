.PHONY: setup test lint bigram transformer
setup:
	python -m pip install -e '.[dev]'
test:
	pytest
lint:
	ruff check .
bigram:
	python labs/03_language_modeling/train_bigram.py
transformer:
	python labs/04_transformer/train_tiny_transformer.py

