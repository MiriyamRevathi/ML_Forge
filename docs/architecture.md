# Platform Architecture

MLForge is organized into four runtime layers. Flask routes translate HTTP requests into validated service calls. Services coordinate storage and machine-learning operations. The `ml` package owns data transformation, training, prediction, monitoring, and lifecycle decisions. Utilities provide bounded file access, logging, metrics, and shared validation.

## Request flow

A request enters through a blueprint in `routes/`, is validated at the boundary, and is passed to a service. Services return serializable dictionaries or domain objects. The route chooses an HTML template or JSON response. Machine-learning work stays below the service layer so the same operations can be reused by tests, scripts, and future workers.

## Storage boundary

The local data directories are treated as an artifact store. Metadata is JSON, tabular datasets are CSV, models are joblib artifacts, and operational reports are JSON. Generated outputs are ignored by Git. This keeps a clone reproducible while allowing a local installation to accumulate experiments and predictions.

## Extension rule

New lifecycle behavior should be added to the owning `ml` module first, exposed through a service, and then wired into a route or command. Avoid putting model decisions in templates or HTTP handlers.
