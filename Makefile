.PHONY : run-checks
run-checks :
	ruff check .
	ruff format --check .
	pytest -v --color=yes tests/

.PHONY : format
format :
	ruff check --fix .
	ruff format .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
