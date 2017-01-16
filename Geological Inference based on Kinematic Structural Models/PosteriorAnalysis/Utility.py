import sys; sys.path.append("C:/Users/lebobcrash/Documents/GitHub/pynoddy/")
import numpy as np
import json
import pynoddy.history
import pynoddy.output
import pynoddy.experiment
import pandas as pn


def calc_fucking_foliations(interface_df):
    test_sorted = interface_df.sort_values("formation")
    _foliation_col_names = ["X", "Y", "Z", "azimuth", "dip", "polarity", "formation"]
    foliations = pn.DataFrame(columns=_foliation_col_names)

    for i,row in enumerate(test_sorted.values):
        #print i,row
        if row[3] == "Layer3" or row[3] == "Layer5":
            if i != len(test_sorted.values)-1:
                if row[3] == test_sorted.values[i+1][3]:
                    dx = row[0] - test_sorted.values[i+1][0]
                    dz = row[2] - test_sorted.values[i+1][2]

                    dip = np.rad2deg(np.arctan(dz/float(dx)))
                    #print dip
                    foliations.loc[len(foliations)] = [np.abs((row[0] + test_sorted.values[i+1][0])/2),
                                                       0,
                                                         np.abs((row[2] + test_sorted.values[i+1][2]) / 2),
                                                         -90,
                                                         dip,
                                                         1.0,
                                                         row[3]]
    return foliations


def noddy_extract_lh(block, x_pos, y_pos, layers):
    storage = []
    for x in x_pos:
        for y in y_pos:
            for layer in np.unique(block[x,y,:]):
                if int(layer) in layers:
                    z = np.where(block[x, y, :] == layer)[0][0]
                    if int(z) != 0:
                        storage.append([x, y, z, "Layer" + str(int(layer))])
                else:
                    pass
    return storage


def pynoddy_assign_priors(prior_dict, ex, verbose=None):
    """
    Function to assign prior draws to respective pynoddy event properties automatically.

    :param prior_dict: dictionary of prior parameters {parameter_key: {mean: m, stdev: sigma, pymc: pymc.Distr object}}
    :param ex: pynoddy.experiment object
    :return: Nothing, sets prior parameters in pynoddy.experiments
    """
    # TODO: Add Rotation event support
    # iterate over every event in the experiment
    for event in ex.events:
        if verbose == 1:
            print event, ex.events[event]
        # if stratigraphy event
        if type(ex.events[event]) is pynoddy.events.Stratigraphy:
            for i, layer in enumerate(ex.events[event].layers):
                for prior in prior_dict.keys():
                    if prior[5] == str(ex.events[event].layers[i].properties["Unit Name"]):
                        ex.events[event].layers[i].properties["Height"] = prior_dict[prior]["pymc"]

        # if fold event
        elif type(ex.events[event]) is pynoddy.events.Fold:
            for prior in prior_dict.keys():
                if prior[5:] in ex.events[event].properties:
                    ex.events[event].properties[prior[5:]] = prior_dict[prior]["pymc"]

        # if unconformity event
        elif type(ex.events[event]) is pynoddy.events.Unconformity:
            for prior in prior_dict.keys():
                
                if prior[13:] in ex.events[event].properties:
                    if verbose == 1:
                        print prior_dict[prior]["pymc"]
                    ex.events[event].properties[prior[13:]] = prior_dict[prior]["pymc"]


def check_adjacency(layer_a, layer_b, topology):
    """
    Check for adjacency of the two layers A and B. Returns True if connected, else False.
    Function should hold true even for very weird model realisations with unexpected edges.
    layerA,layerB: str like "003" for layer #3
    """
    adjacency = None
    for node in topology.graph.node.keys():
        if layer_a in node:
            # print topology.graph.edge[node].keys()
            for edge in topology.graph.edge[node].keys():
                if layer_b in edge:
                    # print topology.graph.edge[node][edge]
                    # print "Node ",layerA," and ",layerB," share an edge!"
                    adjacency = True
    if adjacency is True:
        return True
    else:
        return False


def save_priors(name, prior_dict):
    """
    Saves priors from *prior_dictionary* into "*name*_priors.json" file.
    
    Structure example:
    -------------------------
    prior_dictionary = {
        "prior_name": {
            "mean": 450.,       # Parameters names can vary between different types of distributions!
            "stdev": 40.,
            "pymc": pymc.Distribution
        }
    }
    -------------------------
    """
    with open(name + "_priors.json", "w") as fp:
        json.dump(prior_dict, fp)


def load_priors(file_name):
    """Loads priors from "*name*_priors.json" file."""
    with open(file_name, "r") as fp:
        priors = json.load(fp)
    return priors


# ------------------------------------------------------------------------------------------------------------
# deprecated
# ------------------------------------------------------------------------------------------------------------


def save_parameters(name, parameters):
    f = open(name + ".txt", "w")

    for entry in parameters:
        f.write(str(entry) + "\n")
    f.close()


def save_posteriors(params, pymc_run):
    for entity in params:
        np.save(str(entity) + "_posterior", pymc_run.trace(entity)[:])
        print str(entity) + "_posterior saved successfully."
