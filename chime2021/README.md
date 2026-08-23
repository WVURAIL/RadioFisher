# CHIME Overview configuration

This directory retains the two scientific inputs needed to identify the
as-built CHIME configuration used for the CHIME instrument overview forecast:

- `experiments_CHIME.py`, the experiment dictionary; and
- `array_config/nx_CHIME_800.dat`, the processed baseline-density table for
  the four-cylinder, 1,024-feed layout.

The old forecast, DESI, plotting, and baseline-generation frontends were
removed from the active 1.0 tree. They were Python 2 campaign scripts with
fixed paths and missing or generated inputs, not supported package APIs. Use
Git history if their methodology is needed, then port the relevant calculation
to the current `radiofisher` package and record all resolved inputs.

`experiments_CHIME.py` is deliberately importable from a source checkout and
uses the tracked baseline table above. It is not installed as part of the
`radiofisher` wheel.

The forecast was produced through combined efforts by Tianyue Chen and Simon
Foreman.
