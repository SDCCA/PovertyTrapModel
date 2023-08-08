import xarray as xr 

def data_collection(agent_graph, npath, epath, ndata = ['all'], edata = ['all'], mode = 'xarray'):
    # data_collection - collects data from agents for each time-step  
    for nprop in ndata:
        node_property_collector(agent_graph, npath, nprop, mode)
    
    for eprop in edata:
        edge_property_collector(agent_graph, epath, eprop, mode)


def node_property_collector(agent_graph, npath, nprop, mode):
    if os.environ["DGLBACKEND"] == "pytorch":
        if mode == 'xarray':
            agent_data_instance = xr.Dataset()
            for prop in nprop:
                agent_data_instance = agent_data_instance.assign(prop=(['n_agents','time'], agent_graph.ndata[prop].numpy()))
            # agent_data = xr.concat([agent_data, agent_data_instance], dim='time')
            agent_data_instance.to_zarr(npath, append_dim='time')
        else:
            raise NotImplementedError("Only 'xarray' mode currrent available")
    else:
        raise NotImplementedError("Data collection currently only implemented for pytorch backend")
        

def edge_property_collector(agent_graph, epath, eprop, mode):
    if os.environ["DGLBACKEND"] == "pytorch":
        if mode == 'xarray':
            edge_data_instance = xr.Dataset(coords=dict(
                                            source=(["n_edges"], agent_graph.edges()[0]),
                                            dest=(["n_edges"], agent_graph.edges()[1]),
                                            time=1))
            for prop in eprop:
                edge_data_instance = edge_data_instance.assign(prop=(['n_edges','time'], agent_graph.edata[prop].numpy()))
            edge_data_instance.to_zarr(epath, append_dim='time')
        else:
            raise NotImplementedError("Only 'xarray' mode currrent available")
    else:
        raise NotImplementedError("Data collection currently only implemented for pytorch backend")