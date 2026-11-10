"""Browser front end for Quantize.

Serves a local web UI for building a case from dropdowns, validating it,
running it, and reading the fitted structure back. Needs nothing beyond the
package's own dependencies -- no Flask, no Qt, no display server -- so it works
over SSH and inside containers.

The UI does not reimplement the pipeline. It builds the same config dict a YAML
input file produces and hands it to the same validate/run path the CLI uses, so
there is exactly one way a case is interpreted. A case built in the browser can
be downloaded as YAML and run with ``python -m cli run case.yaml``; a YAML file
written by hand can be loaded into the form. Neither path is a second
implementation of the other.
"""
