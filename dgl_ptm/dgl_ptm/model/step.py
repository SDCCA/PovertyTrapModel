#!/usr/bin/env python
# coding: utf-8

# step - time-stepping for the poverty-trap model

from dgl_ptm.agentInteraction import trade_money
from dgl_ptm.network import local_attachment, link_deletion, global_attachment
from dgl_ptm.agent import agent_update
from dgl_ptm.model import data_collection

def step(agent_graph, timestep, params):
    '''
        step - time-stepping module for the poverty-trap model

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            timestep: Current time step
            params: List of user-defined parameters

        Output:
            agent_graph: Updated agent_graph after one step of functional manipulation
    '''

    #Wealth transfer
    trade_money(agent_graph, method = params['wealth_method'])
    
    #Link/edge manipulation
    local_attachment(agent_graph)
    link_deletion(agent_graph, del_prob = params['del_prob'])
    global_attachment(agent_graph, ratio = params['ratio'])
    
    #Update agent states
    agent_update(agent_graph)

    #Data collection and storage
    data_collection(agent_graph, timestep = timestep, npath = params['npath'], epath = params['epath'], ndata = params['ndata'], 
                    edata = params['edata'], mode = params['mode'])
    