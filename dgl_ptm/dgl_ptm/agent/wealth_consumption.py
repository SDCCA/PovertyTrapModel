def wealth_consumption(model_graph, method='pseudo_consumption'):
    # Calculate wealth consumed
    if method == 'pseudo_wealth_consumption':
        _pseudo_wealth_consumption(model_graph)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'pseudo_consumption'")
    

def _pseudo_wealth_consumption(model_graph):
    model_graph.ndata['wealth_consumption'] = model_graph.ndata['k']*1./3.