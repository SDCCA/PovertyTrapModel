import torch
import numpy as np

def income_generation(model_graph, params=None, method='pseudo_income_generation'):
    # Calculate income generated   
    if method == 'income_generation':
        _income_generator(model_graph, params)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'income_generation'")
    

def _income_generator(model_graph, params):
    gamma = params['tech_gamma']
    cost = params['tech_cost']
    model_graph.ndata['income'],model_graph.ndata['tech_index'] = torch.max((model_graph.ndata['alpha'][:,None]*model_graph.ndata['wealth'][:,None]**gamma - cost), axis=1)
