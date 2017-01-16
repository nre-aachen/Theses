# Posteriors Class to handle Posterior Outputs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from scipy.stats.kde import gaussian_kde

import json
import pymc

import sys

sys.path.append("C:/Users/lebobcrash/Documents/GitHub/pynoddy/")

import pynoddy.history
import pynoddy.output
import pynoddy.experiment


class PosteriorAnalysis:
    """Handles PYMC run posteriors for pynoddy graben simulations."""

    def __init__(self, db_name, sim_type):
        # self.parameters_filename = parameters
        self.sim_type = sim_type  # pynoddy or gempy
        self.db = pymc.database.hdf5.load(db_name)

        # get trace names
        self.trace_names = self.db.trace_names[0]

        self.prior_trace_names = []
        for entry in self.trace_names:
            if "Metropolis" not in entry and "_model" not in entry and "deviance" not in entry and "like" not in entry:
                self.prior_trace_names.append(entry)


        if sim_type is "gempy":
            self.blocks = self.db.gempy_model.gettrace()
        elif sim_type is "pynoddy":
            self.blocks = self.db.pynoddy_model.gettrace()
        #elif sim_type is "both":
        #    self.gempy_sections = self.db.gempy_model.gettrace()
        #    self.pynoddy_sections = self.db.pynoddy_model.gettrace()

        if len(np.shape(self.blocks)) is 3:
            self.blocks = np.expand_dims(self.blocks, axis=2)

        # load posteriors
        # self.posteriors = {}
        # for entry in self.parameters:
        #    self.posteriors[entry] = np.load(entry+"_posterior.npy")

        self._layers = np.unique(self.blocks[:])
        self._n_layers = len(self._layers)
        self._n_iter = len(self.blocks)
        self.x_extent = len(self.blocks[0, :, 0, 0])
        self.y_extent = len(self.blocks[0, 0, :, 0])
        self.z_extent = len(self.blocks[0, 0, 0, :])
        self._NaN = -9999

    #def noddy_plot_average_trace_model(self):



    def plot_traces(self):
        fig, ax = plt.subplots(nrows=len(self.prior_trace_names), ncols=1, figsize=(8,len(self.prior_trace_names)*3.2))
        for r,tr in enumerate(self.prior_trace_names):
            ax[r].plot(self.db.trace(tr)[:])
            ax[r].set_title(tr+" trace")

    def plot_section(self, n, y=0):
        ax = None
        if ax == None:
            ax = plt.axes()

        ax.imshow(self.blocks[n, :, y, :].T, origin="lower")

    def browse_sections(self):
        from IPython.html.widgets import interact
        interact(self.plot_section, n=(0, self._n_iter - 1, 1))

    def compute_entropy(self):
        """
        Calculates the per-voxel entropy of the model section.
        Return: numpy ndarray
        """
        lith_count = np.zeros_like(self.blocks[0:self._n_layers])


        for x in range(self.x_extent):
            for z in range(self.z_extent):
                for y in range(self.y_extent):
                    for layer in range(self._n_layers):
                        lith_count[layer, x, y, z] = np.count_nonzero(self.blocks[:, x, y, z] == layer + 1)

        lith_entropy = np.zeros_like(lith_count[:, :, :, :])

        for i in range(self._n_layers):
            pm = lith_count[i, :, :, :] / self._n_iter
            lith_entropy[i] = -(pm * np.log2(pm))
            lith_entropy[i] = np.nan_to_num(lith_entropy[i])

        entropy = np.sum(lith_entropy[:, :, :, :], axis=0)
        self.entropy_voxels = entropy
        self.entropy_total = np.sum(self.entropy_voxels) / (self.x_extent * self.z_extent * self.y_extent)

        print("Per-voxel IE has been saved into self.entropy_voxels and the total IE into self.entropy_total.")

    def plot_entropy(self, y=0):
        if not hasattr(self, 'entropy_voxels'):
            return "Better calculate the information entropy first."

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(self.entropy_voxels[:,y,:].T, origin="lower", interpolation="nearest", cmap="viridis")
        plt.title("Information Entropy")
        plt.xlabel("x [Voxels]")
        plt.ylabel("z [Voxels]")
        plt.colorbar(shrink=0.77);
        plt.show()

    def extract_layer_thickness(self, x):
        """
        Input:
            x = int : x-coordinate
        Return:
            np.array of thickness values
        """
        thickness_dict = {}
        for layer in self._layers:
            thickness = np.array([])
            for i in range(self._n_iter):
                thickness = np.append(thickness, np.count_nonzero(self.blocks[i, x, :] == layer))
            thickness_dict[str(int(layer))] = thickness
        return thickness_dict

    def extract_layer_height(self, x):
        height_dict = {}
        for layer in self._layers:
            height = np.array([])
            for i in range(self._n_iter):
                temp = np.where(self.blocks[i, x, 0, :] == layer)
                if len(temp[0]) == 0:
                    pass  #height = np.append(height, self._NaN)
                else:
                    height = np.append(height, temp[0][0])
            height_dict[int(layer)] = height

        return height_dict

    def extract_layer_height_dict(self, x_pos, drop=None):
        """
        x_pos: list of x-coordinates at which to extract the layer heights
        drop: int or list of ints for layer numbers to drop
        """
        dictionary = {x: self.extract_layer_height(x) for x in x_pos}

        if drop is not None:
            for x in x_pos:
                if type(drop) is list:
                    for entry in drop:
                        dictionary[x].pop(entry, None)
                else:
                    dictionary[x].pop(drop, None)

        return dictionary

    def kde_dict(self, dictionary):
        d = {}
        for x_pos in dictionary.keys():
            d[x_pos] = {}
            for layer in dictionary[x_pos].keys():
                if len(dictionary[x_pos][layer]) > 0 and len(np.unique(dictionary[x_pos][layer])) > 1:
                    d[x_pos][layer] = self.kde(dictionary[x_pos][layer])

        #return {x_pos: {layer: self.kde(dictionary[x_pos][layer]) for layer in dictionary[x_pos].keys()} for x_pos in
        #        dictionary.keys()}

        return d

    def _kde_sklearn(self, x, x_grid, bandwidth):
        """Kernel Density Estimation with Scikit-learn"""
        kde_skl = KernelDensity(bandwidth=bandwidth)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])

        return (np.exp(log_pdf), kde_skl, x_grid)

    def kde2(self, list_or_array, bw=None):
        """Method to make a kernel density estimation from a 1-D numpy array."""
        hist_array = np.array(list_or_array)

        # bandwidth cross-validation
        if bw == None:
            grid = GridSearchCV(KernelDensity(),
                                {'bandwidth': np.linspace(0.1, 10., 50)},
                                cv=20)  # 20-fold cross-validation
            grid.fit(hist_array.reshape(-1, 1))
            bw = grid.best_params_["bandwidth"]

        x_grid = np.linspace(np.min(hist_array), np.max(hist_array), 500)
        pdf = self._kde_sklearn(hist_array, x_grid, bw)

        return pdf

    def kde(self, data):
        """Gaussian KDE estimation using scipy.stats.kde. Input: histogram data (list or array)"""
        return gaussian_kde(np.array(data))

    def like(self, data, value):
        """Returns the log probability of the given value from the KDE of the given data (could be slow)."""
        return np.log(self.kde(data).evaluate(value))[0]

    def priors_load(self):
        """Loads priors settings from json file into dictionary."""
        with open(self.name + "_priors.json", "r") as fp:
            self.priors = json.load(fp)

    def priors_sample(self, n):
        """Generates n samples of all priors"""
        self.prior_samples = {}
        for key in self.priors:
            if str(self.priors[str(key)]["type"]) == "Normal":
                self.prior_samples[str(key)] = [pymc.Normal(str(key), self.priors[str(key)]["mean"],
                                                            1. / np.square(self.priors[str(key)]["stdev"])).random() for
                                                i in range(n)]
            elif str(self.priors[str(key)]["type"]) == "DiscreteUniform":
                self.prior_samples[str(key)] = [pymc.DiscreteUniform(str(key), self.priors[str(key)]["lower"],
                                                                     self.priors[str(key)]["upper"]).random() for i in
                                                range(n)]
            else:
                print("Distribution type not supported.")
                break

    def priors_plot_all(self):
        """Generates basic histogram plots for all priors."""
        prior_fig, prior_axes = plt.subplots(nrows=len(self.priors), ncols=1, figsize=(6, len(self.priors) * 4.5))

        i = 0
        for key in self.prior_samples:
            prior_axes[i].hist(self.prior_samples[str(key)], label=str(key))
            prior_axes[i].legend()
            i += 1
