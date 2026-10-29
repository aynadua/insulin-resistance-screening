.PHONY: help setup lint test data cohort evaluate train report all clean

PY ?= python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"

lint:  ## static checks
	ruff check src tests

test:  ## unit tests
	pytest

data:  ## download the NHANES cycles (~40 MB)
	irisk data

cohort:  ## build the analysis cohort and print the participant flow
	irisk cohort

evaluate:  ## score the model against the clinical indices
	irisk evaluate

train:  ## fit on every cycle and export a serving artefact
	irisk train

report:  ## collect metrics and figures into reports/
	irisk report

all: data evaluate train report  ## full reproduction

clean:
	rm -rf models reports/*.json reports/*.png reports/RESULTS.md
