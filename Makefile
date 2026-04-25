.PHONY: install run run-cli test lint clean

install:
	pip install -r requirements.txt

run:
	python web_app.py

run-cli:
	cd src && python main.py

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ web_app.py --max-line-length=120 --select=E9,F63,F7,F82

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
