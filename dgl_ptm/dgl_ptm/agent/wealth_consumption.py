import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


def wealth_consumption(model_graph, model_params=None, method='pseudo_consumption'):
    # Calculate wealth consumed
    if method == 'pseudo_consumption':
        _pseudo_wealth_consumption(model_graph)
    elif method == 'fitted_consumption':
        _fitted_wealth_consumption(model_graph)
    elif method == 'bellman_consumption':
        _bellman_wealth_consumption(model_graph,model_params)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'pseudo_consumption','fitted_consumption', or 'bellman_consumption'")
    

def _pseudo_wealth_consumption(model_graph):
    model_graph.ndata['wealth_consumption'] = model_graph.ndata['wealth']*1./3.
    
def _fitted_wealth_consumption(model_graph):
    model_graph.ndata['wealth_consumption'] = 0.64036047*torch.log(model_graph.ndata['wealth'])

def _bellman_wealth_consumption(model_graph, model_params):
    # ToDo: Implement bellman wealth model
    raise NotImplementedError(
            "Bellman wealth consumption model is currently not available. Please await future updates.")