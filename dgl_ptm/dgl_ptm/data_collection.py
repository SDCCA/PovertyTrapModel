import xarray as xr 
import dgl

def data_collection(agent_graph, npath='./agent_data', epath='./edge_data', ndata = ['all'], edata = ['all'], mode = 'xarray'):
    # data_collection - collects data from agents for each time-step  

    if ndata == ['all']:
        ndata = list(agent_graph.node_attr_schemes().keys())
    if edata == ['all']:
        edata = list(agent_graph.edge_attr_schemes().keys())

    node_property_collector(agent_graph, npath, ndata, mode)
    edge_property_collector(agent_graph, epath, edata, mode)


def node_property_collector(agent_graph, npath, ndata, mode):
    if os.environ["DGLBACKEND"] == "pytorch":
        if mode == 'xarray':
            agent_data_instance = xr.Dataset()
            for prop in ndata:
                _check_nprop_in_graph(agent_graph, prop)
                agent_data_instance = agent_data_instance.assign(prop=(['n_agents','time'], agent_graph.ndata[prop].numpy()))
            # agent_data = xr.concat([agent_data, agent_data_instance], dim='time')
            agent_data_instance.to_zarr(npath, append_dim='time')
        else:
            raise NotImplementedError("Only 'xarray' mode currrent available")
    else:
        raise NotImplementedError("Data collection currently only implemented for pytorch backend")
        

def edge_property_collector(agent_graph, epath, edata, mode):
    if os.environ["DGLBACKEND"] == "pytorch":
        if mode == 'xarray':
            edge_data_instance = xr.Dataset(coords=dict(
                                            source=(["n_edges"], agent_graph.edges()[0]),
                                            dest=(["n_edges"], agent_graph.edges()[1]),
                                            time=1))
            for prop in edata:
                _check_eprop_in_graph(agent_graph, prop)
                edge_data_instance = edge_data_instance.assign(prop=(['n_edges','time'], agent_graph.edata[prop].numpy()))
            edge_data_instance.to_zarr(epath, append_dim='time')
        else:
            raise NotImplementedError("Only 'xarray' mode currrent available")
    else:
        raise NotImplementedError("Data collection currently only implemented for pytorch backend")
    
def _check_nprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.node_attr_schemes().keys():
        raise ValueError(f"{prop} is not a node property. Please choose from {agent_graph.node_attr_schemes().keys()}")
    
def _check_eprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.edge_attr_schemes().keys():
        raise ValueError(f"{prop} is not an edge property. Please choose from {agent_graph.edge_attr_schemes().keys()}")