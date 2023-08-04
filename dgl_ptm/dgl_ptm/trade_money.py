import os
import torch 
import dgl 
import dgl.function as fn

def trade_money(agent_graph, method: str):
    """ Trades money between the different connected agents based on 
        wealth (k) and savings propensity (lambda). 
        
        Two methods are provided for wealth exchange: 
        1 - Random weighted wealth transfer to all connected neigbours
        2 - Choose one random neighbour for transfering wealth 

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            method: String with method choice of 'weighted_transfer' or 'singular_transfer'

        Output:
            agent_graph.ndata['delta_inc']: Adds node attribute 'delta_inc' with amount of
                wealth transfered from neighbouring agent nodes to self in this time-step.

    NOTE: Assumes that the following properties are available already:
        k, lambda, w (for 'weighted_transfer'), zeros, ones, total neighbour count
    NOTE: All edges are bidirected with uniform weights 'w'

    TODO: Rename variables as per Thijs' updates on notebook
    """
    # Calculating disposable wealth
    agent_graph.ndata['disp_wealth'] = agent_graph.ndata['lambda']*agent_graph.ndata['k']

    # Transfer of wealth
    if method == 'weighted_transfer':
        _weighted_transfer(agent_graph)
    elif method == 'singular_transfer':
        _singular_transfer(agent_graph)
    else:
        raise ValueError("Incorrect method received. \
                         Method needs to be either 'weighted_transfer' or 'singular_transfer'")
    
def _weighted_transfer(agent_graph):
    """
        Weighted transfer of wealth from each agent (node) to every connected 
        neighbour agent based on pre-defined edge weights.
    """

    # Sum all incoming weights
    agent_graph.ndata['tot_wgt'] = torch.zeros(g.num_nodes(),1)
    agent_graph.update_all(fn.u_add_e('tot_wgt','w','tot_wgt'), fn.sum('tot_wgt', 'tot_wgt'))

    # Calculating outgoing weight %s
    agent_graph.apply_edges(fn.e_div_u('w','tot_wgt','per_wgt'))

    # Wealth transfer amount on each edge
    agent_graph.apply_edges(fn.e_mul_u('per_wgt','disp_wealth','trfr_wealth'))

    # Sum total income from wealth exchange
    agent_graph.update_all(fn.v_add_e('zeros','trfr_wealth','delta_inc'), fn.sum('delta_inc', 'delta_inc'))

def _singular_transfer(agent_graph):
    """
        Singular transfer of wealth from each agent (node) to one randomly
        selected, connected neighbour.
    """

    # Subsample graph to one random edge per node
    graph_subset = dgl.sampling.sample_neighbors(agent_graph, agent_graph.nodes(), 1, edge_dir='out', copy_ndata = True)
    
    # Calculate incoming wealth for each agent in subgraph
    graph_subset.ndata['delta_inc'] = torch.zeros(agent_graph.num_nodes(),1)
    graph_subset.update_all(fn.u_add_v('disp_wealth','zeros','delta_inc'), fn.sum('delta_inc', 'delta_inc'))

    # Update wealth delta in agent graph
    agent_graph.ndata['delta_inc'] = graph_subset.ndata['delta_inc']
