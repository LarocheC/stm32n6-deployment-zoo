# Lab notebook

Prose only. No code, no imports from here into `zoo/`.

One file per model, named for its recipe id. What goes in: what was tried, what
broke, what the numbers were, what to try next, and the dead ends — especially
the dead ends, since the whole point of the zoo is that a failure explained is
worth more than a success unexplained.

The rule that keeps this from metastasising into the core: **anything in `lab/`
that gets used twice moves into `zoo/` with a test.** A one-off snippet lives
here as a fenced block in prose; the second time it is needed, it becomes a
module.
