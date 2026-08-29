# Testing Contract

The test suite exercises the application factory, route registration, dataset validation, exploratory analysis, pipeline execution, model registry transitions, predictions, monitoring, and retraining. Run it from the repository root with `pytest`.

Tests use temporary storage where they create artifacts. Production data directories are never a source of fixtures. A focused test should verify the public service or route contract and should assert both successful output and the relevant validation failure.

Before a release, run `pytest`, `python -m compileall -q app.py config.py ml routes services utils`, and build the container with `docker build -t mlforge:latest .`.
