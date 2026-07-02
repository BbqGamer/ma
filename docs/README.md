Generating the docs
----------

Use [mkdocs](http://www.mkdocs.org/) structure to update the documentation.

The repository now has two main experiment tracks:

- `ma_thesis/` for Gaussian-continuation regression experiments tracked with MLflow.
- `coarse-to-fine-curriculum/` for the second-stage coarse-to-fine image-classification
  experiments tracked with W&B and used heavily in the later thesis appendices.

Build locally with:

    mkdocs build

Serve locally with:

    mkdocs serve
