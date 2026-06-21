# Agent Conventions

## Git

- **Do not commit to `master`/`main`.** Always create or switch to a feature branch before committing. The default branch is reserved for merged, reviewed work.
- **Never push.** Do not run `git push` (or any other remote-publishing command) under any circumstances. Commit locally only — the human handles all pushes.

## Testing

- **Favour property-based tests (`quickcheck`) where applicable.** When a primitive has invariants that hold across the whole input space — a native path matching its scalar oracle, algebraic properties (additivity, commutativity, lane independence), identity elements — assert the property over random inputs rather than relying solely on hand-picked vectors. Keep a few hand-rolled tests for hand-computed known values and as readable documentation, but let properties cover the breadth.
