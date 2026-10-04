# Git workflow

- Commit completed changes as you work, without waiting for a separate commit request.
- Keep commits small, focused, and logically complete: one coherent change per commit.
- Include relevant tests and documentation with the change they support. Separate unrelated changes into their own commits.
- Run appropriate checks before committing, and keep each commit buildable and testable.
- Follow the repository's Conventional Commits style: `type(scope): concise imperative summary`, for example `feat(dap): select shader entry point by name and stage`.
- Use types such as `feat`, `fix`, `refactor`, `test`, `docs`, and `chore`; choose a scope that identifies the affected component when useful.
- Stage only files or hunks belonging to the intended change, preserving unrelated work.
