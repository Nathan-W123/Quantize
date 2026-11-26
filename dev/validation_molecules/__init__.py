"""Transcribed validation-molecule entries, one module per molecule.

Each module defines ``ENTRY``, a plain dict with the fields of
dev.monofluoro_references.ReferenceMolecule plus provenance (per-species source
and note, a second equilibrium structure where one is published). Next to each
is ``<name>.numbers.json``, which gives every transcribed number with its table,
row, column and exact source text line. dev/validation_references.py turns the
entries into ReferenceMolecule objects; that module, not these, is what the
benchmark imports.
"""
