# RobotTaskAssignment

A minimal sandbox for simulating robots and goals on a 2D plane and saving quick scatter plots of their positions. Useful as a starting point for task assignment experiments or simple visualization.

## Features

- World model with `Entity` objects (robots and goals)
- Distance and neighborhood computation based on a sensing range
- Fast plotting that saves timestamped images to `plots/`

## Project Layout

- `model/mark1.py` — Core model with `World` and `Entity` classes, simple runner.
- `plots/` — Output directory for generated images.
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.9+ recommended
- Packages listed in `requirements.txt` (`numpy`, `matplotlib`, `seaborn`)

## Setup

Optionally use a virtual environment to keep things isolated.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

Run the basic simulation (no plot by default):

```bash
python3 model/mark1.py
```

Enable plotting explicitly via Fire flag:

```bash
python3 model/mark1.py --plot true
```

Optional params:

```bash
python3 model/mark1.py --m 10 --n 10 --seed 42 --plot true
```

When plotting is enabled, the script will:

1) Create `M` goals and `N` robots at random positions in `[0,1]^2`.
2) Compute pairwise distances and binary neighbor flags based on each entity's `sensingrange`.
3) Save a scatter plot under `plots/` with a timestamped filename.

Notes:

- Goals are plotted as red circles; robots as green plus markers.
- Each run is stochastic. For reproducibility you can set a seed at the top of `model/mark1.py`, e.g.:

  ```python
  import numpy as np
  np.random.seed(42)
  ```

## Customization

- Modify `M` and `N` at the top of `model/mark1.py`.
- Change `sensingrange` by passing a different value when constructing `Entity` objects.
- Extend `World` with your own dynamics or assignment logic.

## Troubleshooting

- If plots are not appearing on screen, that is expected — the script saves images to disk and closes the figure.
- If `matplotlib` complains about backends, ensure you are using the non-interactive save workflow above (no `plt.show()` is used).
- If installation fails, upgrade `pip` and try again:

  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Next Steps

- Add a simple CLI (e.g., `--robots`, `--goals`, `--seed`).
- Implement actual task assignment strategies and visualize outcomes.
- Add tests around distance and neighborhood computations.
