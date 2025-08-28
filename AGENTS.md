# Repository Guidelines

## Project Structure & Module Organization
- `model/mark1.py`: Core logic with `World` and `Entity`, plus a simple runner that generates plots.
- `plots/`: Timestamped PNGs saved by the runner (artifacts, not source).
- `requirements.txt`: Python dependencies (`numpy`, `matplotlib`, `seaborn`).
- Tests are not yet present; see Testing Guidelines to add them.

## Build, Test, and Development Commands
- Setup environment:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run locally:
  - `python3 model/mark1.py` — creates random robots/goals and saves a scatter plot under `plots/`.
- Reproducibility tip:
  - Add `np.random.seed(42)` near the top of `model/mark1.py` while experimenting.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, max sensible line lengths, clear names.
- Naming: classes `PascalCase` (`World`, `Entity`); functions/variables `snake_case`; constants `UPPER_SNAKE_CASE`.
- Prefer type hints for new/modified functions; add short docstrings for public methods.
- Keep modules small and focused; avoid mixing plotting, I/O, and algorithms without clear separation.

## Testing Guidelines
- Framework: use `pytest` if adding tests.
- Layout: create `tests/` with files like `test_world.py`, `test_distances.py`.
- Conventions: name tests `test_*` and assert behaviors (e.g., neighbor flags, distance symmetry).
- Commands: after installing `pytest`, run `pytest -q` from repo root.

## Commit & Pull Request Guidelines
- Commits: present log is brief; prefer imperative, scoped messages. Example: `feat: add neighbor matrix check` or `fix(world): handle duplicate entity ids`.
- PRs: include a concise description, linked issues, run/verify steps, and before/after plot screenshots if visuals change.
- Keep PRs small, focused, and with updated docs when behavior or usage changes.

## Security & Configuration Tips
- Generated images can be large; consider adding `plots/` to `.gitignore` for heavy iterations.
- Filenames include spaces/colons; this is fine on POSIX. For cross-platform workflows, consider a safe format in future changes (e.g., `"%Y%m%d-%H%M%S"`).
