from mesa import Agent, Model
#from mesa.time import RandomActivation
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm #this was altered to accomodate new requirements
import seaborn as sns
import networkx as nx
import os, sys
from itertools import product
import math
import statistics
import cmath
'''VMG notes: 
- I have not thought about why link_deletion varies from the notebook version

- Due to version differences, there are now problems with random choice 
(see https://github.com/python/cpython/issues/100805)
because of this, I have changed all instances of random.choice(self.model.nodes) and random.choice(self.nodes)
to random.choice(list(self.model.nodes)) and random.choice(list(self.nodes)), respectively.
There should be a better way, but until the network mechanisms have been finalized, 
I think this will serve as a sufficient patch.

-The progress bar is not functioning, perhaps because of my tqdm import edit.

-Sigma, risk averseness for the isoelastic utility function, is now heterogeneous.
'''


#0<gamma_L<gamma_H<1
#gamma_L = 0.3
#gamma_H = 0.45
#fixed_cost = 0.45
sigma = 1.5
beta = 0.95
delta = 0.08
theta = 0.8
g = 1
p_ga = random.random()#for global attachment
p_ld = random.random()#for link deletion
p_delta = 0.3#for local attachment

TechTable = {#contains values for 0:gamma 1:cost
# VMG things get stuck in a while loop if the gamma is less than 0.3 (tried 0.2)
# not sure yet if/how this will be problematic 
    "low":   [0.3,  0   ],
   # "medium":[0.35, 0.25],
    "high":  [0.45, 0.45]}


#global function that calculates the weight of the edge, args: the 2 nodes (agent class objects)
#this is the homophily edge weight formula
def Edge_Weight(node1,node2, b, a):
        try:
             weight = 1+math.exp(a*((node1.k-node2.k)-b))
        except OverflowError:
             weight = float('inf')
        return 1/weight  
    
def calculating_k_c(agent, gamma, E_t, time):
        a1 = pow(agent.k,gamma) 
       
        #k_t+1 = theta*(alpha*k_t^gamma - C_t + (1-delta)*k_t)
        k_new = theta*(agent.alpha*a1-agent.consum + (1-delta)*agent.k)
        #print("New k = ", k_new)

        slope = gamma*agent.alpha*pow(agent.k, gamma -1) + 1 - delta - 1/theta
        #print("Slope = ", slope)
    
        #k_t+1^(gamma-1)
        a2 = pow(k_new,(gamma-1)) 

        #beta*E*theta*(alpha*gamma*k_t+1^(gamma-1)+(1-delta))
        e1 = pow(beta, time - 1)*E_t*theta*(agent.alpha*gamma*a2 + (1-delta)) 

        #(beta*E_t*theta*(alpha*gamma_H*a2 + (1-delta)))^(1/sigma)
        e2 = pow(e1, (1/agent.sigma))
        
        #c*sigmathroot(beta*E_t*theta*(alpha*gamma_H*a2 + (1-delta)))^(1/sigma)
        con = agent.consum * e2
        #print("Calculated consumption :", con)

        return k_new, con, slope
    
def isocline(agent): #Eq 3.7
        con_cond = agent.alpha*pow(agent.k, TechTable[agent.tec][0]) + (1-delta)*agent.k - agent.k/theta
        return con_cond   

def introduce_noise(agent, c_cond):
        if(c_cond <= 0):
            print("Invalid c_cond:",c_cond)
            return 0
        if(1 <= c_cond - agent.k):
            print("Invalid c_cond:",c_cond,"versus k:",agent.k)
            return agent.k
        # introduce stochastic noise to the isocline, conforming to boundary conditions:
        # * consumption cannot be less than 0
        # * consumption cannot be more than the current capital
        # * consumption cannot be more than the isocline.
        # * stochastic noise is at more 1
        #con = c_cond - random.random()
        #while(con > agent.k or con < 0):
        #    con = c_cond - random.random()
        min_val = max(c_cond - agent.k, 0)
        max_val = min(c_cond, 1)
        con = c_cond - random.uniform(min_val, max_val)
        return con
    
        
class MoneyAgent(Agent):
    
    def __init__(self, unique_id, model):
        
        super().__init__(unique_id, model)
        self.k = (capital[unique_id]) #initial stock of wealth
        self.lamda = round(random.uniform(0.1,1),1) #saving propensity
        while (self.lamda == 1):
            self.lamda = round(random.uniform(0.1,1),1)    
        self.alpha = alpha[unique_id]#human capital 
        self.sigma = sigma #risk averseness
        self.tec = 'NA'
        self.income = 0 #initialising income
        self.income_generation() #finding income corresponding to the human capital,
                                 #needed here to set the initial consumption
        self.fronts = 0 #for resetting micawber frontier 
        con_cond = isocline(self)
        #self.consum = isocline(self)
        #if(self.consum < 0):
            #self.consum = 0.1

        self.slope = TechTable[self.tec][0] * self.alpha * pow(self.k, TechTable[self.tec][0] - 1) + 1 - delta - 1/theta
        #print("Checkpoint slope",unique_id)

        if(self.slope > 0): #small k_t
            #print("1st quadrant",self.slope)
            if(con_cond > 0 and con_cond < self.k):
                self.consum = con_cond
            else:
                self.consum = introduce_noise(self, c_cond=con_cond)
        else:
            #print("4th quadrant",self.slope)
            if(con_cond > 0 and con_cond < self.k):
                self.consum = con_cond
            else:
                self.consum = introduce_noise(self, c_cond=con_cond)
    
        self.model.agents.append(self)

        
        
    #function that decides income based on the type of technology
    def income_generation(self): 
        #print("Generating income")
        f = {}
        for i in TechTable.keys(): #in the end, they may each need their own tech table
            entry = self.alpha * pow(self.k, TechTable[i][0]) - TechTable[i][1]
            f[i] = entry
        
        self.fronts = f

        #VMG a technology is chosen based on maximizing income
        self.tec = max(f, key=f.get)
        self.income = f.get(self.tec)
   
            
    
    #function that updates the capital and consumption for the next time step    
    def income_updation(self):
        #print("Income updation")
        #finding expected value of income at each time step
        e_t = [a.income for a in self.model.agents] #is this k or f(alpha,k)?
        E_t = statistics.mean(e_t)
        k = self.k
        alpha = self.alpha
        consum = self.consum
        #print("Agent:{}  Tec: {}".format(self.unique_id, self.tec))
        #print("Old k = {}, alpha = {} " .format(k, alpha))
        #print("mean : ", E_t)
            
        k_new, con, slope = calculating_k_c(self, TechTable[self.tec][0], E_t, self.model.time)
        self.k = k_new
            
        c_cond = isocline(self)
            #print("c_cond = ", c_cond)

        if(slope > 0):
            #print("1st quadrant II")
            if(con <= c_cond and con < self.k):
                self.consum = con
            else:
                self.consum = introduce_noise(self, c_cond)
        else:
            #print("4th quadrant II")
            if(con > c_cond and con < self.k):
                self.consum = con
            else:
                self.consum = introduce_noise(self, c_cond)

        
        #print("Old C:", consum)   
        #print("New Consum :", self.consum)
    
    #finding neighbor nodes for the purpose of making an edge/connection
    def neighbors(self):
        neighbors_nodes = list(nx.all_neighbors(self.model.G,self.unique_id))
        neighbors = []
        for node in neighbors_nodes:
            for agent in self.model.agents:
                if(agent.unique_id == node):
                    neighbors.append(agent)
        return neighbors
    
     #function used to trade/communicate     
    def give_money(self): 
        #print('Give Money from agent: ',self.unique_id)
        b = self.model.b
        a = self.model.a
        neighbors = self.neighbors()
        epsilon = random.random()
        if len(neighbors) > 1 :
            other = self.random.choice(neighbors)
            while(other.unique_id == self.unique_id):
                other = self.random.choice(neighbors)  
            w = self.model.G[self.unique_id][other.unique_id]['weight'] 
            if(w >= random.random()): 
                xi = self.income
                xj = other.income
                delta_income = (1-self.lamda)*(xi - epsilon*(xi + xj))
                xi_new = xi - delta_income
                xj_new = xj + delta_income
                other.income = xj_new
                self.income = xi_new
                for neighbor in neighbors:
                    self.model.G[self.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(self,neighbor,b, a)
                other_neighbors = other.neighbors()
                for neighbor in other_neighbors:
                    if(neighbor.unique_id != other.unique_id):
                        self.model.G[other.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(other,neighbor,b, a)
        
    def LocalAttachment_v1(self): 
        #print('Local Attachment V1')
        b = self.model.b
        a = self.model.a
        node1 = random.choice(list(self.model.nodes))
        node2 = random.choice(list(self.model.nodes))
        count = 0 #to avoid an infinite loop when all agents have already made links with each other
        while(self.model.G.has_edge(node1,node2)==True and count <5):
            node2 = random.choice(list(self.model.nodes))
            node1 = random.choice(list(self.model.nodes))
            count +=1
        for agent in self.model.agents:
            if(agent.unique_id == node1):
                node1_a = agent
            if(agent.unique_id == node2):
                node2_a = agent
        self.model.G.add_edge(node1,node2,weight = Edge_Weight(node1_a,node2_a, b, a))
        
        
    #1. select nodes i and j, with a probability proportional to the weight (wij) between them.
    #2. j selects a neighbour with a probability proportional to the weight between them such that there is no edge between 
    #i and k. 
    #3. a link is made between i and k with edge weight w0 (here, w0 = 1) and all edge weights are increased by wr(wr-calculated 
    #new edge weight between i and j)
    #
    #
    # From Paper [1]:
    # A randomly chosen node i (its degree is k; with k_i≠0) chooses one of its neighbors j [later: filtered by having similar capital]
    #   with the probability proportional to w_{ij} which stands for the weight of the link between nodes i and j.
    # Ten node j chooses one of its neighbors [later: filtered by having similar capital] but i, say k, randomly
    #   with probability proportional to w_{jk} and
    # if nodes i and k are not connected, they are connected with probability p_Δ with a link of weight w_0.
    # In addition, all the involved [later: existing] links increase the weights by w_r, whether a new link is created or not.
    #
    # Q: what is p_Δ? [A: 0.02]
    # Q: what is w_0?
    # Q: what is w_r? [A: 1]
    #
    # Later:
    # "Indeed, the link reinforcement in LA plays here a key role [...]"
    # So neither p_Δ or w_r should be set to 0.
    def LocalAttachment_v2(self):
        #print("Local Attachment V2")
        b = self.model.b
        a = self.model.a
        #print("LA done")
        links_nodes = list(self.model.G.edges(data= 'weight'))
        #print('Link nodes=', links_nodes)
        edge_weight_sum = sum(k for i, j,k in links_nodes)
        #print('Weight sum = ', edge_weight_sum)
        edge_weights = []
        for edge in links_nodes:
            edge_weights.append(edge[2])
        #print('Edge weights = ', edge_weights)
        edge_prob = []
        for edge_w in edge_weights:
            edge_prob.append(edge_w/edge_weight_sum)
        #print('Edge prob = ', edge_prob)
        arr_pos = [i for i in range(len(links_nodes))]
        pos = np.random.choice(arr_pos, p = edge_prob)
        chosen = links_nodes[pos]
        #print("Chosen nodes:", chosen[0:2])
        node1 = chosen[0]
        node2 = chosen[1]
        node1_a = next((agent for agent in self.model.agents if agent.unique_id == node1), None)
        node2_a = next((agent for agent in self.model.agents if agent.unique_id == node2), None)
        #print("Keeping node1 as:", node1)
        #print("Keeping node2 as:", node2)
        #finding neighbors of node2
        neighbors = [n for n in self.model.G.neighbors(node2)]
        #print("Neighbors of {} are:{}".format(node2,neighbors))
        if(len(neighbors)>1):
            #print("Finding 3rd node")
            links_nodes = list(self.model.G.edges(node2, data= 'weight'))
            #print('Link nodes=', links_nodes)
            edge_weight_sum = sum(k for i, j,k in links_nodes)
            #print('Weight sum = ', edge_weight_sum)
            edge_weights = []
            for edge in links_nodes:
                edge_weights.append(edge[2])
            #print('Edge weights = ', edge_weights)
            edge_prob = []
            for edge_w in edge_weights:
                edge_prob.append(edge_w/edge_weight_sum)
            #print('Edge prob = ', edge_prob)
            arr_pos = [i for i in range(len(links_nodes))]
            pos = np.random.choice(arr_pos, p = edge_prob)
            chosen = links_nodes[pos]
            #print("Chosen nodes:", chosen[0:2])
            for node in chosen[0:2]: #because the 3rd element is weight
                if(node!= node2):
                    node3 = node
                    #print('3rd node = ', node3)
                    if(self.model.G.has_edge(node1,node3)==False and random.random()>p_delta):
                        node3_a = next((agent for agent in self.model.agents if agent.unique_id == node3), None)
                        self.model.G.add_edge(node1,node3,weight = Edge_Weight(node1_a,node3_a, b, a))
                    
    
   #links are deleted randomly at every time step
    #Notes (tvl):
    # Runs once per AGENT step.
    # So, up to 1 edge is removed each model step.
    # The condition compares a random number to the global value, which is also a random number...
    # Assuming this condition is passed, it chooses a random node
    # and then it has 5 chances to pick a random node and hope that it is connected to that node.
    # Note that the same node can be chosen twice, so it could check whether the node is attached to itself.
    # It also does not use the network when selecting the second node.
    #   This means that if the number of agents is large relative to their network size, the chances of
    #   accidentally selecting two nodes that are connected is very small.
    #
    # Suggestion:
    # Replace by picking a random node and a random neighbor of that node.
    #
    #
    # From Paper [1]:
    # Each link is removed from the system with probability p_d at each time step. (Time is measured in sweeps).
    #
    # Q: what is p_d? [A: 0.005]
    # Presumably a time step is a model step.
    # However, in the paper, the features are static, so the only thing that changes is the network (edges and edge weights).
    def Link_Deletion(self):
        #print('Deletion')
        if(random.random()>p_ld):
            node1 = random.choice(list(self.model.nodes))
            node2 = random.choice(list(self.model.nodes))
            count = 0
            while(self.model.G.has_edge(node1,node2)==False and count<5):
                node2 = random.choice(list(self.model.nodes))
                count +=1
            if(count !=5):
                self.model.G.remove_edge(node1,node2)
        #print('deletion done')
                    
    def step(self):
        print('This should not be running! Please check!')
        ##if(self.k > 0):
        #self.income_updation()
        #self.give_money()
        ##self.LocalAttachment_v1()
        #self.LocalAttachment_v2()
        #self.Link_Deletion()
        #self.income_generation() 
        
        
        
class BoltzmannWealthModelNetwork(Model):
    """A model with some number of agents."""

    def __init__(self,b, a,N=100): #N- number of agents

        self.N = N
        self.b =b
        self.a = a
        self.agents = []
        self.gini = 0
        self.time = 0
        self.count_GA = 0
        self.G = nx.barabasi_albert_graph(n=N, m = 1)
        nx.set_edge_attributes(self.G, 1, 'weight') #setting all initial edges with a weight of 1
        self.nodes = np.linspace(0,N-1,N, dtype = 'int') #to keep track of the N nodes   
        
        #self.schedule = RandomActivation(self)
        self.schedule = StagedActivation(self, ['income_updation','give_money','LocalAttachment_v2','Link_Deletion','income_generation'], shuffle=True, shuffle_between_stages=True)
        self.datacollector = DataCollector(model_reporters = {"Gini": 'gini'},agent_reporters={"k_t":'k','income':'income',
                                           'Fronts':'fronts', 'consumption':'consum','lamda':'lamda','alpha':'alpha', 'technology':'tec' })       
        for i, node in enumerate(self.G.nodes()):
            agent = MoneyAgent(i, self)
            self.schedule.add(agent)
           
        self.running = True
        self.datacollector.collect(self)
        
    #Notes (tvl):
    # Runs once per MODEL step.
    # So, up to 1 edge is added each model step.
    # The condition compares a random number to the global value, which is also a random number...
    # The same node can be chosen twice, so it could be attached to itself.
    #
    # Suggestion:
    # Replace by shuffle(list) and look for the first successive pair that does not have an edge?
    #   So, suffle and then connect (0,1) or (1,2) or (2,3) etc.
    #
    #
    # From Paper [1]:
    # A node is selected at random, its degree is k.
    # With probability δ_{0,k}+(1−δ_{0,k})p_r it is connected with a new link of weight w_0 to a randomly chosen node.
    # In the unlikely event that the two nodes are already connected, a new target node is chosen.
    #
    # Q: what is δ_{0,k}? [A: Kronecker delta of 0 and k (i.e. node degree): 0 if 0 \neq k; 1 if 0 = k]
    #    [Note, this answer is debatable, because the notation δ_{0,k} is not defined, but δ(i,j) is.]
    # Q: what is p_r? [A: 0.001]
    # Q: what is w_0?
    #
    # From Paper [1]:
    # More specifcally, in a GA step, the node i chooses randomly a node, say j, sharing the same trait for the feature f in the network. [i.e. similar capital]
    # If the node i has any neighbor sharing the same trait for the feature f, then the link between i and j is created with probability p_r.
    # Otherwise, the link is created with probability one.
    # The weight of the created link is given as w0.
    #
    # Q: what is p_r? [A: 0.001]
    def Global_Attachment(self):
        if(random.random()>p_ga):
            #print("Global Attachment no: {}".format(self.count_GA))
            self.count_GA+=1
            node1 = random.choice(list(self.nodes))
            node2 = random.choice(list(self.nodes))
            while(self.G.has_edge(node1,node2)==True):
                node2 = random.choice(list(self.nodes))
                node1 = random.choice(list(self.nodes))
            #adding the edge node1-node2
            #first: find the class object corresponding to the node
            for agent in self.agents:
                if(agent.unique_id == node1):
                    node1_a = agent
                if(agent.unique_id == node2):
                    node2_a = agent
            #a and b are pre-determined, a is the homophily parameter and b is the characteristic distance
            self.G.add_edge(node1,node2,weight =  Edge_Weight(node1_a,node2_a, self.b, self.a))
        
    def compute_gini(self):
        agent_wealths = [agent.k for agent in self.schedule.agents]
        x = sorted(agent_wealths)
        B = sum(xi * (self.N - i) for i, xi in enumerate(x)) / (self.N * sum(x))
        return 1 + (1 / self.N) - 2 * B
    
    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    # From Paper [1]:
    # Sequential update is applied, first GA, then LA to the nodes and then LD to the links. 
    #
    # In each GA and LA step, a feature f of the focal node i is randomly chosen from F features and it can make links only
    #   to the nodes sharing the same trait for the feature f, i.e., only to the nodes j satisfying σ_j^f = σ_i^f.
    #   [in our case, there is only one feature: current wealth. However, the similarity of this feature should infuence the selection probability for both GA and LA.]
    def run_model(self, n):
        for i in tqdm(range(n)):
            #print("Step:", i+1)
            self.time = i+1
            self.step()
            self.Global_Attachment()
            self.gini = self.compute_gini()
            

#Notes (tvl):
# Why are these model globals not defined at the top, with the agent globals?
# Should the model and agent be split into separate sources?
N = 50
steps = 125
b = 35
a = 0.69
alpha = np.random.normal(loc = 1.08, scale = 0.074, size = N) 
capital = np.random.uniform(low = 0.1, high = 10, size = N)

model = BoltzmannWealthModelNetwork(b, a,N)
model.run_model(steps)
model_df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
agent_df.reset_index(level=1, inplace = True)
agent_df.to_csv("V2SE_{}Agents_{}Steps.csv".format(N,steps))
model_df.to_csv("V2SE_Model_{}Agents_{}Steps.csv".format(N,steps))

# [1] Structural transition in social networks: The role of homophily, authors-authors-authors, place