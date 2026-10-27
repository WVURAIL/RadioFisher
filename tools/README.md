# Documentation tools

The purpose of `generate_call_graph.py` is to draw `docs/radiofisher_functions.svg`,
the function-level diagram of the forecast in the main README. Its colors come from
`wvu_palette.py`. Neither script is part of the forecast library. From the
repository root, the diagram is regenerated with `python tools/generate_call_graph.py`.

Boxes and labels are written by hand in the generator. When the forecast changes,
the labels are updated in the generator and the SVG is regenerated. The labels
distinguish between RadioFisher's Simpson integration and RFIsher's segmented
quadrature, and between the bias projection B and the parameter bias
inverse(F_fit) B.
