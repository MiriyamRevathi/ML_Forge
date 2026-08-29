# Release Checklist

- Confirm the working tree contains source, documentation, manifests, and tests only.
- Install from `requirements.lock` in a clean virtual environment.
- Run the complete `pytest` suite and compile the application packages.
- Verify `/diagnostics` reports healthy dependencies and writable storage.
- Build the Docker image and start a disposable container.
- Review model and dataset metadata for accidental local paths or sensitive values.
- Push the release branch and merge it through the hosting provider so review history remains discoverable.

Generated artifacts stay outside the release commit and are recreated by the sample-data or application workflows.
