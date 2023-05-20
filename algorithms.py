import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

class OptimizationAlgorithm:
    def __init__(self, grid:gpd.GeoDataFrame, resolution):
        grid['X'] = grid.geometry.centroid.x
        grid['Y'] = grid.geometry.centroid.y
        centroids = grid.copy(deep=True)
        centroids['geometry'] = centroids.geometry.centroid
        self.grid = grid
        self.centroids = centroids
        self.resolution = resolution
        self.buffer = None
        self.shops = gpd.GeoDataFrame()
        self.points = {}
    def score(self, indexes):
        shops = self.centroids.iloc[indexes]
        buffer = shops.geometry.buffer(self.buffer+1)
        if len(indexes) > 1:
            buffer = buffer.unary_union
        intsc_centr = self.centroids[self.centroids.geometry.intersects(buffer)]
        return intsc_centr.ludnosc.sum()
    def plot_map(self):
        fig, ax = plt.subplots()
        self.grid.plot(ax=ax, column='ludnosc', zorder=0)
        for key, line in self.points.items():
            ax.plot(*zip(*line), color='black', linewidth=1, zorder=1)
        self.shops.plot(ax=ax, color='red', zorder=2)
        self.shops.geometry.buffer(self.buffer).plot(ax=ax, facecolor='none', edgecolor='red', zorder=3)
        return fig, ax


class LocationAnnealing(OptimizationAlgorithm):
    def __init__(self, grid:gpd.GeoDataFrame, resolution):
        super().__init__(grid, resolution)
        self.obj_vals = []
        self.probs = []
        self.temps = []
    def get_largest_cells(self, n=6):
        largest_cells = self.centroids.sort_values(by=['ludnosc'], ascending=False).head(n)
        self.shops = largest_cells
        coords = [(obj.X, obj.Y) for i, obj in largest_cells.iterrows()]
        for i, coord in enumerate(coords):
            self.points[str(i+1)] = self.points[str(i+1)] + [coord]
        return largest_cells.index.to_list()
    def get_random_cells(self, n=6):
        indexes = np.random.randint(0, self.centroids.shape[0], size=n)
        random_cells = self.centroids.iloc[indexes]
        self.shops = random_cells
        coords = [(obj.X, obj.Y) for i, obj in random_cells.iterrows()]
        for i, coord in enumerate(coords):
            self.points[str(i+1)] = self.points[str(i+1)] + [coord]
        return random_cells.index.to_list()
    def move_shop(self, neighbourhood, method='random'):
        shops = self.shops.copy(deep=True)
        choice_idx = np.random.choice(self.shops.index.to_list())
        choice = self.centroids.iloc[choice_idx]
        range = neighbourhood * self.resolution
        neighbors:pd.DataFrame = self.centroids[
            (self.centroids.X <= choice.X + range) & (self.centroids.X >= choice.X - range) & (self.centroids.Y <= choice.Y + range) & (self.centroids.Y >= choice.Y - range)
            & ~self.centroids.index.isin(shops.index.to_list())
        ]
        if method == 'steep':
            best = 0
            result = None
            for i, neighbor in neighbors.iterrows():
                objective = self.score([i])
                if objective > best:
                    best = objective
                    result = neighbor
        elif method == 'random':
            neigh_choice = np.random.choice(neighbors.index.to_list())
            result = self.centroids.iloc[neigh_choice]
        elif method == 'greedy':
            curObj = self.score([choice_idx])
            result = None
            for i, neighbor in neighbors.iterrows():
                objective = self.score([i])
                if objective > curObj:
                    result = neighbor
                    break
            if result is None:
                neigh_choice = np.random.choice(neighbors.index.to_list())
                result = self.centroids.iloc[neigh_choice]
        
        for key, value in self.points.items():
            prev = value[-1]
            chosen = (choice.X, choice.Y)
            if prev == chosen:
                to_change = {"line": key, "point": (result.X, result.Y)}
                break
        shops.drop(choice_idx, inplace=True)
        shops = pd.concat([shops, gpd.GeoDataFrame(result.to_dict(), index=[result.name], columns=['ludnosc', 'geometry', 'X', 'Y'])])
        return shops, to_change
    def accept_or_reject(self, new_version, to_change):
        new_score = self.score(new_version.index.to_list())
        print(f'{round(self.objective, 3)} vs. {round(new_score, 3)}')
        if self.objective < new_score:
            self.shops = new_version.copy(deep=True)
            self.objective = new_score
            self.obj_vals.append(new_score)
            self.probs.append(self.probs[-1] if len(self.probs) > 0 else 1)
            self.points[to_change["line"]] = self.points[to_change["line"]] + [to_change["point"]]
            return True
        else:
            prob = np.exp2(-(self.objective - new_score) / self.temp)
            self.probs.append(prob)
            #print(f'Diff: {round((self.objective - new_score), 6)} Prob: {prob}')
            accept:bool = np.random.random(size=1) < prob
            if accept:
                #print('ACCEPTED')
                self.shops = new_version.copy(deep=True)
                self.objective = new_score
                self.obj_vals.append(new_score)
                self.points[to_change["line"]] = self.points[to_change["line"]] + [to_change["point"]]
            else:
                #print('REJECTED')
                self.obj_vals.append(self.objective)
            return accept
    def run(self, init='highest', move_choice='random',
            neighbourhood=2, n_shops=6, buffer=1500,
            objective=None, n_evals=None, prop_rejected=0.9,
            start_temp=0.001, temp_mult=None, temp_substr=None):
        self.temp = start_temp
        self.buffer = buffer
        self.points = {str(key):[] for key in range(1, n_shops+1)}
        if init == 'highest':
            init_indexes = self.get_largest_cells(n_shops)
        elif init == "random":
            init_indexes = self.get_random_cells(n_shops)
        else:
            raise AttributeError("Wrong method of initialization")
        self.objective = self.score(init_indexes)
        reject_count = 0
        evals = 0
        while True:
            new_version, to_change = self.move_shop(neighbourhood, move_choice)
            accept = self.accept_or_reject(new_version, to_change)
            self.temps.append(self.temp)
            reject_count = reject_count + 1 if not accept else reject_count
            evals += 1
            if self.temp > 0.005 * start_temp and temp_mult is not None:
                self.temp *= temp_mult
            elif self.temp > 0.005 * start_temp and temp_substr is not None:
                self.temp -= temp_substr
            
            if objective is not None:
                if self.objective >= objective:
                    status = 1
                    print('Objective achieved')
                    break
            if n_evals is not None:
                if evals > n_evals:
                    status = 2
                    print('Number of evaluations done')
                    break
            if prop_rejected is not None:
                if reject_count / evals > prop_rejected and evals > 50:
                    status = 3
                    print('Large proportion of rejected permutations')
                    break
        print('DONE')
        return self, status
    def plot_objective(self):
        plt.plot(self.obj_vals)
        plt.title('Objective change')
    def plot_probs(self):
        plt.plot(self.probs, color='black', linewidth=0.5)
        plt.title('Probability change')
    def plot_temp(self):
        plt.plot(self.temps, color='red')
        plt.show()
        plt.close()

class LocationEvolution(OptimizationAlgorithm):
    def __init__(self, grid:gpd.GeoDataFrame, resolution:int, results:list[gpd.GeoDataFrame], buffer:int=1500):
        super().__init__(grid, resolution)
        self.buffer = buffer
        self.population = results
        self.scores = [self.score(opt.index.to_list()) for opt in self.population]
        self.mins = []
        self.means = []
        self.maxs = []
    def choose_parents(self) -> list[gpd.GeoDataFrame]:
        while True:
            samples = list(np.random.choice(range(0, len(self.population)), size=2, replace=False))
            parents = [self.population[i] for i in samples]
            if not bool(set(parents[0].index) & set(parents[1].index)):
                break
        return parents
    def crossover(self):
        parents = self.choose_parents()
        indexes1 = list(np.random.choice(range(0, parents[0].shape[0]), size=3, replace=False))
        indexes2 = [i for i in range(0, parents[0].shape[0]) if not i in indexes1]
        offsprings = [
            pd.concat([parents[0].loc[parents[0].index[indexes1]], parents[1].loc[parents[1].index[indexes2]]]),
            pd.concat([parents[0].loc[parents[0].index[indexes2]], parents[1].loc[parents[1].index[indexes1]]]),
        ]
        return offsprings
    def replace(self, offsprings:'list[gpd.GeoDataFrame]'):
        worst = np.argsort(self.scores)[:2]
        for idx in sorted(worst, reverse = True):
            del self.population[idx]
            del self.scores[idx]
        self.population += offsprings
        self.scores += [self.score(kid.index.to_list()) for kid in offsprings]
    def run(self, epochs=100):
        print(f'Min: {min(self.scores)} | Mean: {np.mean(self.scores)} | Max: {max(self.scores)}')
        for i in range(0, epochs):
            offsprings = self.crossover()
            self.replace(offsprings)
            self.mins.append(min(self.scores))
            self.means.append(np.mean(self.scores))
            self.maxs.append(max(self.scores))
            print(f'Min: {min(self.scores)} | Mean: {np.mean(self.scores)} | Max: {max(self.scores)}')
        
        best_idx = np.argsort(self.scores)[-1]
        best = self.population[best_idx]
        self.shops = best
        return best
    def plot_objective(self):
        fig, ax = plt.subplots()
        ax.plot(self.mins, color='red', linewidth=2)
        ax.plot(self.means, color='blue', linewidth=2)
        ax.plot(self.maxs, color='green', linewidth=2)
        plt.show()
        plt.close()
