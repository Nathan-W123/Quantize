# Vendored third-party assets

## 3Dmol-min.js — 3Dmol.js 2.1.0

WebGL molecular viewer, BSD-3-Clause. https://3dmol.csb.pitt.edu/

Vendored rather than loaded from a CDN so the UI keeps working offline, behind
a proxy and inside a container, which is where it mostly runs. A viewer that
silently fails to load is worse than one that is a few hundred kilobytes
larger, and this is the one asset where the alternative — hand-rolling the
lighting in canvas 2D — was visibly not good enough.

Served by `web/server.py` at `/vendor/3Dmol-min.js`.
