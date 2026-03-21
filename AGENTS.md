# AGENTS.md

## Project

RamanKit is a Python library for Raman spectroscopy data processing and analysis.

The project should favor a typed, scientifically explicit API over convenience shortcuts.
Prefer clear domain abstractions, predictable behavior, and reproducible workflows.

## Priority order

When rules pull in different directions, prefer:

- scientific correctness and explicit assumptions over convenience
- clear, stable public APIs over ad hoc shortcuts
- reproducible, traceable transformations over hidden automation
- small, maintainable NumPy/SciPy-first implementations over speculative abstraction

## Architecture principles

- Preserve the current domain model:
  - `Spectrum`
  - `SpectrumCollection`
  - `RamanImage`
- Keep preprocessing explicit and composable through reusable pipeline steps.
- Preserve metadata and provenance whenever data is transformed.
- Prefer immutable-style transformations that return new objects instead of mutating inputs in place.
- Keep plotting as a thin presentation layer, not a computation layer.
- Keep I/O separated from core domain logic.
- Prefer small, composable building blocks over large monolithic helpers.

## Scientific guardrails

- Do not introduce implicit spectral-axis alignment.
- Do not introduce hidden resampling, interpolation, normalization, smoothing, or baseline correction.
- Do not silently change axis units, axis meaning, or metadata semantics.
- Do not hide scientifically meaningful assumptions inside convenience APIs.
- Make all transformations explicit in code and traceable in provenance.

## Code layout

Treat this as the canonical package layout unless there is a clear reason to deviate.

```text
src/ramankit/
├── core/           # Core domain models, metadata, provenance, shared operations
├── preprocessing/  # Atomic preprocessing steps for Raman data
├── pipelines/      # Reusable preprocessing workflows
├── io/             # File loaders, savers, and persistence
├── peaks/          # Peak detection and peak fitting
├── plotting/       # Plotting and visualization helpers
└── synthetic/      # Synthetic Raman data generation
```

## Public API boundary

- Expose public APIs intentionally through `src/ramankit/__init__.py`.
- Keep internal helpers internal unless they are part of a documented Raman workflow.
- New public functions and classes should have a clear Raman use case and predictable return types.

## Implementation preferences

- Prefer extending existing abstractions instead of adding one-off helpers.
- Prefer backend reuse when appropriate (for example `pybaselines`) instead of reimplementing stable algorithms.
- Prefer explicit parameter validation with clear error messages.
- Prefer focused modules with a single responsibility.
- Prefer dataclasses and small result objects when they improve clarity.
- Prefer straightforward implementations first; optimize only where needed and measurable.

## API design rules

- Public APIs should be explicit, typed, and domain-oriented.
- Avoid adding generic helpers unless they clearly support the Raman workflow.
- Avoid boolean-heavy APIs when an explicit object or enum would be clearer.
- Keep naming scientifically precise and consistent across modules.
- Keep behavior consistent across `Spectrum`, `SpectrumCollection`, and `RamanImage` where possible.

## I/O rules

- Keep vendor-specific parsing isolated inside the `io/` layer.
- Preserve raw vendor metadata when possible.
- Prefer normalized metadata fields in addition to raw metadata, not instead of it.
- Do not leak file-format quirks into the core data model.
- Avoid adding format support through ad hoc parsing scattered across the codebase.

## Provenance rules

- Any meaningful transformation should append an appropriate provenance step.
- Provenance entries should describe the operation name, key parameters, and enough context to understand what happened later.
- Do not drop provenance during derived-object creation unless explicitly documented.

## Performance and dependencies

- Do not add heavy runtime dependencies without strong justification.
- Keep the runtime dependency set small and focused.
- Use NumPy/SciPy-first solutions unless a different dependency provides substantial value.
- Consider memory behavior for Raman images and collections.

## Testing

Run these before finishing work:

- `uv run pytest`
- `uv run ruff check .`
- `uv run mypy src`

Testing expectations:

- Add or update tests for every public behavior change.
- Add regression tests for bug fixes.
- Prefer focused unit tests over large opaque integration tests.
- Test both happy paths and validation/error paths.
- When changing scientific behavior, update tests to make the new behavior explicit.

## Documentation

- Update docs when changing public API or workflow behavior.
- Keep README examples aligned with the actual API.
- Prefer small, runnable examples.
- Document assumptions that affect scientific interpretation.

## What to avoid

- Do not weaken type clarity just to reduce boilerplate.
- Do not mix plotting, I/O, and scientific transformations in the same abstraction.
- Do not add hidden magic intended to “help” the user at the cost of correctness.
- Do not introduce architecture that turns RamanKit into a generic catch-all framework.

## When adding new functionality

Before adding a new feature, check:

1. Does it fit RamanKit’s scope as a Raman spectroscopy processing and analysis library?
2. Does it belong in an existing module?
3. Can it be expressed through existing abstractions?
4. Is the behavior explicit and scientifically defensible?
5. Are tests and docs updated accordingly?
