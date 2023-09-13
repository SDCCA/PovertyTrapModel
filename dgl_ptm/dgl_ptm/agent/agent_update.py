from dgl_ptm.agent.income_generation import income_generation
from dgl_ptm.agent.wealth_consumption import wealth_consumption

def agent_update(model_graph):
    '''
    agent_update - Updates the state of the agent based on income generation and money trades
    '''
    model_graph.ndata['wealth'] = model_graph.ndata['wealth'] + model_graph.ndata['net_trade']
    income_generation(model_graph, method = 'pseudo_income_generation')
    wealth_consumption(model_graph, method='pseudo_wealth_consumption')
    model_graph.ndata['wealth'] = model_graph.ndata['wealth'] + model_graph.ndata['income'] - model_graph.ndata['wealth_consumption']