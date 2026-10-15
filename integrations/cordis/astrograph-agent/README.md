# AstroGraph Cordis/Harness adapter — B009 development evidence

This package does not reimplement AstroGraph numerical science in JavaScript. It starts the content-addressed Python capability server and registers profile-specific model tools through a small Cordis compatibility seam.

Presets physically separate diagnosis, development, post-hoc, evidence-curation, confirmatory-sealed and maintainer tool surfaces. `confirmatory` contains no policy-selection tool. `diagnosis` contains no truth/evaluator tool.

The adapter is dependency-free ESM so it can be fresh-loaded in isolation. A production Harness integration should pin the host version and replace only `src/compat.mjs` if its registry API changes.
