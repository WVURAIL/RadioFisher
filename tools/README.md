# Documentation tools

From the repository root, run `python tools/generate_call_graph.py` to regenerate
`docs/radiofisher_functions.svg`. This replaces the former `docs/diagram.py`
location. It is not part of the forecast library.

The diagram is a curated overview, not an automatically extracted call graph.
When the forecast changes, update the labels in the generator and regenerate
the SVG. It distinguishes RadioFisher's Simpson integration from RFIsher's
segmented quadrature and the residual projection from the resulting bias.
