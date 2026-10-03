"""DVC pipeline stages over the existing modules (``dvc.yaml``; D48).

    python -m turbofan.pipeline <validate|cv_select|train_prod|final_eval|register|all> [--smoke]

Importing this package imports nothing heavy: ``--smoke`` must be able to point
``TURBOFAN_PARAMS`` at a merged parameter file before ``turbofan.config`` is first imported.
"""
