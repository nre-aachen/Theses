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

class Posteriors:
    """Handles PYMC run posteriors for pynoddy graben simulations."""
    def __init__(self, parameters, sim_type):
        self.parameters_filename = parameters
        self.sim_type = sim_type # pynoddy or gempy

        # load parameter list
        self.parameters = []
        file = open(self.parameters_filename+'.txt', 'r')
        for line in file:
            key = line.rstrip('\n')
            self.parameters.append(key)
            #= np.load(self.name+"_posterior_"+key+".npy")
        
        # load posteriors
        self.posteriors = {}
        for entry in self.parameters:
            self.posteriors[entry] = np.load(entry+"_posterior.npy")

        self._layers = np.unique(self.posteriors[self.sim_type+"_model"][1])
        self._n_layers = len(self._layers)
        self._n_iter = len(self.posteriors[self.sim_type+"_model"])
        self.x_extent = len(self.posteriors[self.sim_type+"_model"][0,:,0])
        self.z_extent = len(self.posteriors[self.sim_type+"_model"][0,0,:])
        self._NaN = -9999
     
    def plot_section(self, n, ax=None):
        if ax == None:
            ax = plt.axes()

        ax.imshow(self.posteriors[self.sim_type+"_model"][n,:,:])

    def posteriors_browse_sections(self):
        from IPython.html.widgets import interact
        interact(self.posteriors_plot_section, n=(0,self._n_iter - 1,1), cube_size=(10,50,5))

    def posteriors_plot_fault_dip(self, nbins=24):
        plt.hist(self.posteriors["fault_E_dip"], label="Fault E", alpha=0.65, normed=True, bins=nbins)
        plt.hist(self.posteriors["fault_W_dip"], label="Fault W", alpha=0.65, normed=True, bins=nbins)
        plt.legend()
        plt.title("Likelihoods: Fault Dip [degree]")
        plt.show()

    def posteriors_plot_fault_offset(self, nbins=24):
        plt.hist(self.posteriors["fault_E_offset"], label="Fault E", alpha=0.65, normed=True, bins=nbins)
        plt.hist(self.posteriors["fault_W_offset"], label="Fault W", alpha=0.65, normed=True, bins=nbins)
        plt.legend()
        plt.title("Likelihoods: Fault Offset [m]")
        plt.show()

    def posteriors_plot_fault_timing(self):
        plt.hist(self.posteriors["fault_timing"], alpha=0.65, normed=True, width=1)
        plt.xlim(1,3)
        plt.title("Likelihood: Fault Timing")
        plt.show()

    def sections_browse(self):
        """
        Use a slider to browse through all sections. Only works in IPython Notebooks.
        """
        if not "pynoddy_section" in self.posteriors.keys():
            print "No sections calculated."
        else:
            from IPython.html.widgets import interact
            n = len(self.posteriors["pynoddy_section"][:,0,:])
            interact(self._view_image, i=(0,n-1,1))

    def _view_image(self, i):
        """
        Helper function for browse_sections.
        """
        if not "pynoddy_section" in self.posteriors.keys():
            print "No sections calculated."
        else:
            plt.imshow(self.posteriors["pynoddy_section"][i][:,0,:].T, origin="lower", cmap="YlOrRd", interpolation='none')
            plt.title("y-Section of Model #%s" % i)
            plt.show()

    def sections_browse_likelihoods(self, kind):
        """Interactive Likelihood visualization, allowing to slide through every x-position."""
        if not "pynoddy_section" in self.posteriors.keys():
            print "No sections calculated."
        else:
            from IPython.html.widgets import interact

            if kind is "thickness":
                interact(self._view_hist_thick, x=(0,self.x_extent-1,1))
            elif kind is "height":
                interact(self._view_hist_height, x=(0,self.x_extent-1,1))

    def _view_hist_thick(self, x):
        layer_thick = self.sections_like_layer_thickness(x)
        
        for i in range(1,5):
            plt.hist(layer_thick[str(i)], normed=True, label=str(i), alpha=0.65);
        plt.legend()
        plt.title("Likelihoods: Layer Thicknesses [m] at x=%d" % x);
        plt.xlim(0,160)
        plt.ylim(0,0.08)
        plt.show()

    def _view_hist_height(self, x):
        layer_height = self.sections_like_layer_height(x)
        
        for i in range(1,4):
            plt.hist(layer_height[str(i)][layer_height[str(i)]!=self._NaN], normed=True, label=str(i), alpha=0.65);
        plt.legend()
        plt.title("Likelihoods: Layer Height [m] at x=%d" % x);
        plt.xlim(0,220)
        plt.ylim(0,0.08)
        plt.show()

    def sections_entropy(self):
        """
        Calculates the per-voxel entropy of the model section.
        Return: numpy ndarray
        """
        if not "pynoddy_section" in self.posteriors.keys():
            print "No sections calculated."
        else:
            iterations = len(self.posteriors["pynoddy_section"][:])
            lith_count = np.zeros_like(self.posteriors["pynoddy_section"][0:4])

            for x in range(len(self.posteriors["pynoddy_section"][0,:,0,0])):
                for z in range(len(self.posteriors["pynoddy_section"][0,0,0,:])):
                    lith_count[0,x,0,z] = np.count_nonzero(self.posteriors["pynoddy_section"][:,x,0,z]==1)
                    lith_count[1,x,0,z] = np.count_nonzero(self.posteriors["pynoddy_section"][:,x,0,z]==2)
                    lith_count[2,x,0,z] = np.count_nonzero(self.posteriors["pynoddy_section"][:,x,0,z]==3)
                    lith_count[3,x,0,z] = np.count_nonzero(self.posteriors["pynoddy_section"][:,x,0,z]==4)

            lith_entropy = np.zeros_like(lith_count[:,:,0,:])

            for i in range(4):
                pm = lith_count[i,:,0,:]/iterations
                lith_entropy[i] = -(pm * np.log2(pm))
                lith_entropy[i] = np.nan_to_num(lith_entropy[i])

            entropy = lith_entropy[0]+lith_entropy[1]+lith_entropy[2]+lith_entropy[3]
            self.entropy_voxels = entropy
            self.entropy_total = np.sum(self.entropy_voxels)/(self.x_extent * self.z_extent)
            
            print("Per-voxel IE has been saved into self.entropy_voxels and the total IE into self.entropy_total.")



    def sections_plot_entropy(self):
        if not hasattr(self, 'entropy_voxels'):
            return "Better calculate the information entropy first."

        fig = plt.figure(figsize=(8,6))
        plt.imshow(self.entropy_voxels.T, origin="lower", interpolation="nearest", cmap="viridis")
        plt.title("Information Entropy")
        plt.xlabel("x [Voxels]")
        plt.ylabel("z [Voxels]")
        plt.colorbar(shrink=0.77);
        plt.show()

    def sections_plot_section(self, n_plot, figsize=(6,4)):
        """Plot a single y-section.
        
        Input:
            n_plot = int : Number of model interation to plot
        """
        if not "pynoddy_section" in self.posteriors.keys():
            print "No sections calculated."
        else:
            fig = plt.figure(figsize=figsize)
            plt.imshow(self.posteriors["pynoddy_section"][n_plot][:,0,:].T, origin="lower", cmap="YlOrRd")
            plt.title("Section of Model #%s" % n_plot)
            plt.xlabel("x [Voxels]")
            plt.ylabel("z [Voxels]")
            plt.show()

    def sections_like_layer_thickness(self, x):
        """
        Input:
            x = int : x-coordinate
            Optional:
            _n_layers = int : number of layers in model
        Return:
            np.array of thickness values
        """
        if not "pynoddy_section" in self.posteriors.keys():
            print "No sections calculated."
        else:
            thickness_dict = {}

            for layer in range(1,self._n_layers+1):
                thickness = np.array([])
                for i in range(len(self.posteriors["pynoddy_section"][:])):
                    thickness = np.append(thickness, np.count_nonzero(self.posteriors["pynoddy_section"][i,x,0,:] == layer))
                thickness_dict[str(layer)] = thickness

            return thickness_dict

    def sections_like_likelihood_layer_height(self, x):
        if not "pynoddy_section" in self.posteriors.keys():
            print "No sections calculated."
        else:
            height_dict = {}

            for layer in range(1,self._n_layers+1):
                height = np.array([])
                for i in range(len(self.posteriors["pynoddy_section"][:])):
                    temp = np.where(self.posteriors["pynoddy_section"][i,x,0,:] == layer)
                    if len(temp[0]) == 0:
                        height = np.append(height, self._NaN)
                    else:
                        height = np.append(height, temp[0][0])
                height_dict[str(layer)] = height

            return height_dict

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
                        cv=20) # 20-fold cross-validation
            grid.fit(hist_array.reshape(-1,1))
            bw = grid.best_params_["bandwidth"]
            
        x_grid = np.linspace(np.min(hist_array),np.max(hist_array),500)
        pdf = self._kde_sklearn(hist_array, x_grid, bw)

        return pdf

    def kde(self, data):
        """Gaussian KDE estimation using scipy.stats.kde. Input: histogram data (list or array)"""
        return None #gaussian_kde(np.array(data))

    def like(self, data, value):
        """Returns the log probability of the given value from the KDE of the given data (could be slow)."""
        return np.log(self.kde(data).evaluate(value))[0]

    def priors_load(self):
        """Loads priors settings from json file into dictionary."""
        with open(self.name+"_priors.json", "r") as fp:
            self.priors = json.load(fp)

    def priors_sample(self, n):
        """Generates n samples of all priors"""
        self.prior_samples = {}
        for key in self.priors:
            if str(self.priors[str(key)]["type"])=="Normal":
                self.prior_samples[str(key)] =  [pymc.Normal(str(key), self.priors[str(key)]["mean"],1./np.square(self.priors[str(key)]["stdev"])).random()  for i in range(n)]
            elif str(self.priors[str(key)]["type"])=="DiscreteUniform":
                self.prior_samples[str(key)] =  [pymc.DiscreteUniform(str(key), self.priors[str(key)]["lower"],self.priors[str(key)]["upper"]).random()  for i in range(n)]
            else:
                print("Distribution type not supported.")
                break

    def priors_plot_all(self):
        """Generates basic histogram plots for all priors."""
        prior_fig, prior_axes = plt.subplots(nrows=len(self.priors), ncols=1, figsize=(6, len(self.priors)*4.5))

        i = 0
        for key in self.prior_samples:
            prior_axes[i].hist(self.prior_samples[str(key)], label=str(key))
            prior_axes[i].legend()
            i += 1