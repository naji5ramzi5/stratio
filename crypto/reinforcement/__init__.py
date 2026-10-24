"""Reinforcement learning layer for the crypto project.

NOTE: this package must stay import-light (no torch / elegantrl at import time),
because several other modules in the project are imported eagerly and would break
if importing them dragged in heavy dependencies.
"""
