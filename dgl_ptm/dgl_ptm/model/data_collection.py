import xarray as xr 
import dgl
from pathlib import Path

def data_collection(agent_graph, npath='./agent_data', epath='./edge_data', ndata = ['all'], edata = ['all'], timestep, mode = 'xarray'):
    ''' 
        data_collection - collects data from agents and edges for each time-step of the model

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents.
            npath = path to store node data.
            epath = path to store edge data with one file for each timestep.
            ndata = node data properties to be stored. 
                ['all'] implies all node properties will be saved
            edata = edge data properties to be stored. 
                ['all'] implies all edge properties will be saved
            timestep = current timestep to name folder for edge properties
            mode = storage format
                ['xarray'] saves the properties in zarr format with xarray dataset

        Output:

    '''

    if ndata == ['all']:
        ndata = list(agent_graph.node_attr_schemes().keys())
    if edata == ['all']:
        edata = list(agent_graph.edge_attr_schemes().keys())

    _node_property_collector(agent_graph, npath, ndata, mode)
    _edge_property_collector(agent_graph, epath, edata, timestep, mode)


def _node_property_collector(agent_graph, npath, ndata, mode):
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
        

def _edge_property_collector(agent_graph, epath, edata, timestep, mode): 
    if os.environ["DGLBACKEND"] == "pytorch":
        if mode == 'xarray':
            edge_data_instance = xr.Dataset(coords=dict(
                                            source=(["n_edges"], agent_graph.edges()[0]),
                                            dest=(["n_edges"], agent_graph.edges()[1]),
                                            ))
            for prop in edata:
                _check_eprop_in_graph(agent_graph, prop)
                edge_data_instance = edge_data_instance.assign(property=(['n_edges','time'], agent_graph.edata[prop].numpy()))
                edge_data_instance=edge_data_instance.rename_vars(name_dict={'property':prop})
            edge_data_instance.to_zarr(Path(epath)/str(timestep))
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