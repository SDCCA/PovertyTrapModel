
def agent_perception_update(model_graph, params=None, timestep=None, method='default'):
    # Calculate income generated   
    if method == 'default':
        _agent_theta_update(model_graph, params, timestep)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'default'")
    
def _agent_theta_update(model_graph,model_params,timestep):
    #updates agent perception of theta based on observation and sensitivity
    model_graph.ndata['theta'] = model_graph.ndata['theta'] * (1-model_graph.ndata['sensitivity']) + model_params['modelTheta'][timestep] * model_graph.ndata['sensitivity']