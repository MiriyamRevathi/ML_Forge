# Operations Runbook

## Local startup

Create a virtual environment, install `requirements.lock`, generate sample data when needed, and start `python app.py`. The diagnostic page checks imports, writable artifact directories, and registered application components.

## Artifact hygiene

Experiments, models, monitoring reports, pipelines, predictions, logs, and metadata generated during a run belong to the local artifact store. They should not be committed. To reset a development dataset, remove only the generated files under the relevant `data/` subdirectory and regenerate the sample data.

## Failure triage

1. Open `/diagnostics` and record the failing dependency or storage check.
2. Confirm the selected dataset metadata points to an existing CSV.
3. Inspect the most recent log and experiment JSON before retrying training.
4. Validate the model state transition before serving predictions.
5. Run `pytest` after correcting configuration or dependencies.

The application is intentionally local-first: no credentials or remote service configuration is required for the default workflow.
