import numpy as np

import seaborn as sns
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt

from datetime import datetime


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
        assert e.id not in self.entites.keys()
        self.entites[e.id] = e

    def plot(self):
        x = [e.loc[0] for e in self.entities.values()]
        y = [e.loc[1] for e in self.entities.values()]
        
        colors = ['red' if e.type == 'goal' else 'green' for e in self.entities.values()]
        markers = ['o' if e.type == 'goal' else '+' for e in self.entities.values()]

        sns.scatterplot(x=x, y=y, color=colors)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        plt.savefig(f"plots/{stamp}.png")


class Entity:
    def __init__(self, ID, loc=(0,0), etype="robot", sensingrange=10):
        self.id = ID
        self.loc = np.array(loc)
        self.type = etype
        self.sensingrange = sensingrange
        self.distances = {}
        self.neighbors = {}


if __name__ == "__main__":
    goals = [ Entity(ID=_i, loc=np.random.rand(2)) for _i in range(M) ]
    robots = [ Entity(ID=_i+100, loc=np.random.rand(2), etype='goal') for _i in range(N) ]

    world = World(goals + robots)
    world.makedistances()
    world.plot()




