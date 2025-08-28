import numpy as np

import seaborn as sns
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import os

from datetime import datetime
import fire
from typing import Optional


M = 10 # number of goals
N = 10 # number of robots

# maybe modded
assert M==N

# preference matrix
P = np.random.rand(M,N)


class World:
    def __init__(self, entities: list):
        self.entities = {e.id: e for e in entities}
        

    def makedistances(self, ids=None):
        entities = self.entities
        for _idx,this in entities.items():
            for _idy, that in entities.items():
                this.distances[_idy] = np.linalg.norm(this.loc - that.loc)
                this.neighbors[_idy] = 1 if this.distances[_idy] <= this.sensingrange else 0

    def add_entity(self, e):
        # Ensure unique IDs and correct attribute name
        if e.id in self.entities:
            raise ValueError(f"Entity with id {e.id} already exists")
        self.entities[e.id] = e

    def plot(self):
        # Separate robots and goals for clearer plotting
        goals = [e for e in self.entities.values() if e.type == 'goal']
        robots = [e for e in self.entities.values() if e.type == 'robot']

        if goals:
            gx = [e.loc[0] for e in goals]
            gy = [e.loc[1] for e in goals]
            plt.scatter(gx, gy, c='red', marker='o', label='goal')

        if robots:
            rx = [e.loc[0] for e in robots]
            ry = [e.loc[1] for e in robots]
            plt.scatter(rx, ry, c='green', marker='+', label='robot')

        plt.legend()
        plt.title('World entities')
        plt.xlabel('x')
        plt.ylabel('y')

        os.makedirs('plots', exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        plt.savefig(f"plots/{stamp}.png")
        plt.close()


class Entity:
    def __init__(self, ID, loc=(0,0), etype="robot", sensingrange=10):
        self.id = ID
        self.loc = np.array(loc)
        self.type = etype
        self.sensingrange = sensingrange
        self.distances = {}
        self.neighbors = {}

def main(plot: bool = False, m: int = M, n: int = N, seed: Optional[int] = None):
    """Run a simple world simulation.

    Args:
        plot: If True, save a scatter plot to `plots/`.
        m: Number of goals to create.
        n: Number of robots to create.
        seed: Optional RNG seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    assert m == n, "For now, the demo assumes m == n"

    goals = [Entity(ID=_i, loc=np.random.rand(2), etype='goal') for _i in range(m)]
    robots = [Entity(ID=_i + 100, loc=np.random.rand(2), etype='robot') for _i in range(n)]

    world = World(goals + robots)
    world.makedistances()
    if plot:
        world.plot()


if __name__ == "__main__":
    fire.Fire(main)

