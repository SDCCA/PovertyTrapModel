# agent_update - Updates the state of the agent based on income generation and money trades

def agent_update(model_graph):
    model_graph.ndata['k'] = model_graph.ndata['k'] + model_graph.ndata['net_trade']
    income_generation(model_graph, method = 'pseudo_income_generation')
    wealth_consumption(model_graph, method='pseudo_consumption')
    model_graph.ndata['k'] = model_graph.ndata['k'] + model_graph.ndata['income'] - model_graph.ndata['wealth_consumption']