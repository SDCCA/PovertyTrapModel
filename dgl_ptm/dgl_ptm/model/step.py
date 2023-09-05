#!/usr/bin/env python
# coding: utf-8

# step - time-stepping for the poverty-trap model

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
    data_collection(agent_graph, npath = params['npath'], epath = params['epath'], ndata = params['ndata'], 
                    edata = params['edata'], timestep = timestep, mode = params['mode'])
    