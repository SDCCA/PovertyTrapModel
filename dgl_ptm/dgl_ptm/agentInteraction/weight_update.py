import dgl.function as fn
import numpy as np
import torch

def weight_update(agent_graph, a, b):
    """
    Update function to calculate the weight of edges based on the wealth 
    of the connected nodes according to the formula:
            weight = 1/(1 + e^(a*(m(x_i,x_j)-b))) 
    where:
        a = homophily parameter
        b = characteristic distance between the nodes in embedding space
        m(x_i, x_j) = difference in wealth between connected agents
    """
    agent_graph.edata['weight'] = torch.rand(agent_graph.num_edges(),1)
    agent_graph.apply_edges(fn.u_sub_v('wealth','wealth','wealth_diff'))
    agent_graph.edata['weight'] = 1./(1. + np.exp(a*agent_graph.edata['wealth_diff']-b))