from mesa import Agent, Model
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
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import basinhopping



'''VMG notes: 
- I have not thought about why link_deletion varies from the notebook version

- Due to version differences, there are now problems with random choice 
(see https://github.com/python/cpython/issues/100805)
because of this, I have changed all instances of random.choice(self.model.nodes) and random.choice(self.nodes)
to random.choice(list(self.model.nodes)) and random.choice(list(self.nodes)), respectively.
There should be a better way, but until the network mechanisms have been finalized, 
I think this will serve as a sufficient patch.

-The progress bar is not functioning, perhaps because of my tqdm import edit.
Updates:
-Sigma, risk averseness for the isoelastic utility function, is now heterogeneous,
 but this needs a justifiable range and also should never include 1.
-Funcitionality for any number of income technologies has been added.
-Replaced lambda <1 while loop with the upper limit of 0.949



Version 5 notes:
FinalModelScipyV5 is developed from FinalModelScipyV4.py
In this version, adaptation options are introduced. For each adaptation cost, 
there is a corresponding theta difference multiplier. Perfect knowledge of all 
options and their efficacies is assumed. 
The current optimization chooses the maximum value from the bellman equation 
solutions for each available adaptation option.



'''


#0<gamma_L<gamma_H<1
#gamma_L = 0.3
#gamma_H = 0.45
#fixed_cost = 0.45
#sigma = 1.5

random.seed(42)
np.random.seed(42)

g = 1
p_ga = random.random()#for global attachment
p_ld = random.random()#for link deletion
p_delta = 0.3#for local attachment

# global variables for Bellman equation
𝛿 = 0.08
β = 0.95

#Debug
problemAgents=range(500)


TechTable = {#contains values for 0:gamma 1:cost 2:theta
# VMG things get stuck in a while loop if the gamma is less than 0.3 (tried 0.2)
# not sure yet if/how this will be problematic 
# Also important to make sure values are in the correct order
# i.e. that the threshold between medium and high is at a 
# higher k than the threshold between low and medium 
# This can be checked with k_threshold.py
    "low":   [0.3,  0   ],
    "medium":[0.35, 0.15],
    "high":  [0.45, 0.65]}

TechTableArray = np.array([[ 0.3,  0 ],[0.35, 0.15],[0.45, 0.65]])

AdapTable = {#contains values for 0:theta 1:cost (for consideration:effort? type? design life?)
    "none":   [  0, 0   ],
    "good":   [0.5, 0.25],
    "better": [0.9, 0.45]}

AdapTableArray = np.array([[ 0,  0 ],[0.5, 0.25],[0.9, 0.45]])


#global function that calculates the weight of the edge, args: the 2 nodes (agent class objects)
#this is the homophily edge weight formula
def Edge_Weight(node1,node2, b, a):
        try:
             weight = 1+math.exp(a*((node1.k-node2.k)-b))
        except OverflowError:
             weight = float('inf')
        return 1/weight  




def maximize(g, a, b, args):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    Maximize the function g over the interval [a, b].

    The maximizer of g on any interval is
    also the minimizer of -g.  The tuple args collects any extra
    arguments to g.

    Returns the maximum value and the maximizer.
    """

    objective = lambda x: -g(x, *args)
    result = minimize_scalar(objective, bounds=(a, b), method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum

def utility(c, σ, type="isoelastic"):
    if type == "isoelastic":
        if σ ==1:
            return np.log(c)
        else:
            return (c**(1-σ)-1)/(1-σ)

    else:
        print("Unspecified utility function!!!")


def income_function(k,α): 
    #print("Generating income")
    f = []
    for i in TechTable.keys(): #in the end, they may each need their own tech table
        entry = α * k**TechTable[i][0] - TechTable[i][1]
        f.append(entry)
    return max(f)

def adaptation_function(θ,i_a):
    
    for i in AdapTable.keys(): #in the end, they should have their own adaptation table
        if AdapTable[i][1] <= i_a:
            m = AdapTable[i][0]
        else:
            break
    return θ + m * (1-θ)

        

        
def calculate_next_k(agentinfo):
    k,c,i_a,m = agentinfo.k,agentinfo.consum,agentinfo.i_a,agentinfo.m
    #print(f"Old k and c {k,c}")
    if agentinfo.unique_id in problemAgents:
        print(f"Calculating using k={k}, global_theta={global_θ[model.time]}, m={m}, c={c}, i_a={i_a}, sigma={agentinfo.σ}, alpha={agentinfo.α}, percieved theta={agentinfo.θ}")
    k_tplus1 = (global_θ[model.time] + m * (1-global_θ[model.time])) * (k - c - i_a + (1-𝛿) * k)
    #agentinfo.k=k_tplus1
    
    if agentinfo.unique_id in problemAgents:
   
        print(f"k_t+1 after Calculate={k_tplus1}={agentinfo.k}")
        
    #print(f"New k and c {k_tplus1,c_tplus1}\n")
    return k_tplus1






class BellmanEquation:
     #Adapted from: https://python.quantecon.org/optgrowth.html
    def __init__(self,
                 u,            # utility function
                 f,            # production function, I think the piecewise function may need to be reestablished here for accuracy, more thought needed
                 k,            # current state k_t
                 θ,            # given shock factor θ
                 σ,            # risk averseness
                 α,            # human capital
                 i_a,          # adaptation investment
                 m,            # protection multiplier
                 β=β,          # discount factor
                 𝛿=𝛿,          # depreciation factor 
                 name="BellmanNarrowExtended"):

        self.u, self.f, self.k, self.β, self.θ, self.𝛿, self.σ, self.α, self.i_a, self.m, self.name = u, f, k, β, θ, 𝛿, σ, α, i_a, m, name

        # Set up grid
        
        startgrid=np.array([1.0e-7,1,2,3,4,5,6,7,8,9,10,100])
        #startgrid=np.array([0.001,2,4,6,8,10])

        ind=np.searchsorted(startgrid, k)
        self.grid=np.concatenate((startgrid[:ind],np.array([k-1.0e-06, k]),startgrid[ind:]))

        self.grid=self.grid[self.grid>i_a]
        #self.grid=np.concatenate((startgrid,np.array([k-0.1, k-1.0e-02, k, k+1.0e-02, k+0.1])))

        # Identify target state k
        self.index = np.searchsorted(self.grid, k)-1
    
    def value(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, θ, 𝛿, σ, α, i_a, m = self.u, self.f, self.β, self.θ, self.𝛿, self.σ, self.α, self.i_a, self.m

        v = interp1d(self.grid, v_array, bounds_error=False, fill_value="extrapolate")
        
        return u(c,σ) + β * v((θ + m * (1-θ)) * (f(y,α) - c - i_a + (1 - 𝛿) * y))



def update_bellman(v, bell):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    
    The Bellman operator.  Updates the guess of the value function
    and also computes a v-greedy policy.

      * bell is an instance of Bellman equation
      * v is an array representing a guess of the value function

    """
    v_new = np.empty_like(v)
    v_greedy = np.empty_like(v)
    
    for i in range(len(bell.grid)):
        y = bell.grid[i]
        # if y <= 1e-8:
        #     print(f"struggling with {bell.grid} for y={y} for k = {bell.k} and theta={bell.θ}")
        # Maximize RHS of Bellman equation at state y
        
        c_star, v_max = maximize(bell.value, 1e-8, y-bell.i_a, (y, v))
        #VMG HELP! is (1) subtracting i_a and (2) omitting an entire section of the grid necessary/correct
        v_new[i] = v_max
        v_greedy[i] = c_star

    return v_greedy, v_new

def which_bellman(agentinfo):
    feasible=[]
    if agentinfo.unique_id in problemAgents:
        print(f" k= {agentinfo.k} passed to which_bellman")

    for option in agentinfo.adapt:
        if option[1]>=agentinfo.k:
            pass
        else:
            #print(f'working theta = {agentinfo.θ + option[0] * (1-agentinfo.θ)}, i_a= {option[1]}, k= {agentinfo.k}')
            c,v=solve_bellman(BellmanEquation(u=utility, f=income_function, k=agentinfo.k, θ=agentinfo.θ, σ=agentinfo.σ, α=agentinfo.α, i_a=option[1],m=option[0]))
            feasible.append([v,c,option[1],option[0]])
    if agentinfo.unique_id in problemAgents:

        print(feasible)
    best=min(feasible)

    if agentinfo.unique_id in problemAgents:
        print(f"best={best}")
    return best[1],best[2],best[3]

def solve_bellman(bell,
                  tol=1,
                  min_iter=10,
                  max_iter=1000,
                  verbose=False):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    
    Solve model by iterating with the Bellman operator.

    """


    # Set up loop

    v = bell.u(bell.grid,bell.σ)  # Initial condition
    i = 0
    error = tol + 1

    while (i < max_iter and error > tol) or (i < min_iter):
        v_greedy, v_new = update_bellman(v, bell)
        error = np.abs(v[bell.index] - v_new)[bell.index]
        i += 1
        # if verbose and i % print_skip == 0:
        #     print(f"Error at iteration {i} is {error}.")
        v = v_new

    if error > tol:
        print(f"{bell.name} failed to converge for k={bell.k}, α = {bell.α},σ ={bell.σ}, i_a={bell.i_a}, and modified θ = {bell.θ + bell.m * (1-bell.θ)}!")
    elif verbose:
        print(f"Converged in {i} iterations.")
        print(f"Effective k and new c {np.around(bell.grid[bell.index],3),v_greedy[bell.index]}")
        

    return v_greedy[bell.index],v[bell.index]







class MoneyAgent(Agent):
    
    def __init__(self, unique_id, model):
        
        super().__init__(unique_id, model)
        self.k = (capital[unique_id]) #initial stock of wealth
        self.λ = round(random.uniform(0.1,0.949),1)  #VMG replaced lambda <1 while loop
        self.α = alpha[unique_id]#human capital 
        self.σ = round(random.uniform(1,1.9),1)#risk averseness
        self.θ = random.uniform(0.1,1) #percieved theta (cannot be zero or the optimizer breaks because k<) maybe one day this can be spatial?
        self.sensitivity = random.uniform(0,1) #factor controlling tractability of an agent's perception of theta 
        self.tec = "NA"
        #self.γ = "NA"
        #self.tec_cost = "NA"
        self.adapt = AdapTableArray
        self.m = "NA"
        self.i_a = "NA"
        self.income = 0 #initialising income
        self.record_income() #create record of income and tec
        self.fronts = 0 #for resetting micawber frontier(s?) 
        self.consum = 0
        self.initialize_consumption()
        self.money_traded=0
        self.model.agents.append(self)

        


      

    #function that records income and technology applied for timestep
    def record_income(self): 
        
        f = []
        for i in TechTable.keys(): #in the end, they may each need their own tech table
            entry = self.α * self.k**TechTable[i][0] - TechTable[i][1]
            f.append(entry)
            #print(f"{entry} from {self.α}, {self.k} {TechTable[i][0]}, and {TechTable[i][1]}" )    
    
        if self.unique_id in problemAgents:

            print(f)

        #VMG a technology is chosen based on maximizing income
        self.income = max(f)
        self.tec = f.index(self.income)

        #self.γ = TechTableArray[self.tec][0]
        #self.tec_cost = TechTableArray[self.tec][1]
        

        
    def initialize_consumption(self):
        print(f'\nInitializing agent {self.unique_id}')
        self.consum, self.i_a, self.m =which_bellman(self)
        
    
    #function that updates the capital, consumption, investment, and theta multiplier for the next time step    
    def update_capital(self):
        if self.unique_id in problemAgents:
            print(f"\nUpdating capital of agent {self.unique_id} at time step {model.time}")
        self.k = calculate_next_k(self)

    def update_consumption(self):
        if self.unique_id in problemAgents:
            print(f"\nUpdating consumption of agent {self.unique_id} at time step {model.time}")
        self.consum, self.i_a, self.m=which_bellman(self)
        
    
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
    def trade_money(self): 
        self.money_traded=0
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
                xi = self.k
                xj = other.k
                self.money_traded = epsilon * ((1-self.λ) * self.k + (1-other.λ) * other.k)
                delta_money = (1-self.λ) * self.k - epsilon * ((1-self.λ) * self.k + (1-other.λ) * other.k)
                self.k = xi - delta_money
                other.k = xj + delta_money
                for neighbor in neighbors:
                    self.model.G[self.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(self,neighbor,b, a)
                other_neighbors = other.neighbors()
                for neighbor in other_neighbors:
                    if(neighbor.unique_id != other.unique_id):
                        self.model.G[other.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(other,neighbor,b, a)
            if self.unique_id in problemAgents:
                print(f"Money put up for trade by agent {self.unique_id} with agent {other.unique_id} for timestep {model.time} is {self.money_traded}")

    def update_theta(self):
        #print(f'old = {self.θ}')
        self.θ=self.θ * (1-self.sensitivity) + global_θ[model.time] * self.sensitivity
        #print(f'new = {self.θ}')

    def LocalAttachment_v1(self): 
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
    def LocalAttachment_v2(self):
        b = self.model.b
        a = self.model.a
        #print("LA done")
        links_nodes = list(self.model.G.edges(data= 'weight'))
        #print('Link nodes=', links_nodes)
        edge_weight_sum = sum(k for i, j,k in links_nodes)
        #print('Weight sum = ', edge_weight_sum)
        if edge_weight_sum==0:
            print(links_nodes)

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
                    
    def stageA(self):
        #if(self.k > 0):
        self.update_capital()

    def stageB(self):
        self.trade_money()

    def stageC(self):
        self.record_income()
        self.update_consumption()
        self.update_theta()
        #self.LocalAttachment_v1()
        self.LocalAttachment_v2()
        self.Link_Deletion()
     
        
        
        
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
        stage_list=["stageA","stageB","stageC"]
        self.schedule = StagedActivation(self,stage_list,shuffle=True)
        self.datacollector = DataCollector(model_reporters = {"Gini": 'gini', "globe_theta":'globe_theta'},agent_reporters={"k_t":'k','income':'income',
                                           'Fronts':'fronts', 'consumption':'consum','lambda':'λ','alpha':'α', 'percieved_theta':'θ', 'technology':'tec', "i_a":"i_a", "money_traded":"money_traded"})       
        for i, node in enumerate(self.G.nodes()):
            agent = MoneyAgent(i, self)
            self.schedule.add(agent)
           
        self.running = True
        self.datacollector.collect(self)
        
    def Global_Attachment(self):
        if(random.random()>p_ga):
            #print("Global Attachment no: {}".format(self.count))
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
    '''
    def stepA(self):
        self.schedule.stepA()
    def stepB(self):
        self.schedule.stepB()
    def stepC(self):
        self.schedule.stepC()
        # collect data
        self.datacollector.collect(self)
    ''' 
    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in tqdm(range(n)):

            #print("Step:", i+1)
            self.time = i+1
            self.globe_theta = global_θ[self.time]
            self.step()
            self.Global_Attachment()
            self.gini = self.compute_gini()
            
            
            
N =500
steps = 125
b = 35
a = 0.69
alpha = np.random.normal(loc = 1.08, scale = 0.074, size = N) 
capital = np.random.uniform(low = 0.1, high = 10, size = N)
global_θ = np.random.choice([0.1, 0.5, 0.8, 1], N ,p=[0.01 ,0.1, 0.79, 0.1]).tolist()

model = BoltzmannWealthModelNetwork(b, a, N)
model.run_model(steps)
model_df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
agent_df.reset_index(level=1, inplace = True)
agent_df.to_csv("{}Agents_{}StepsScipyV6.csv".format(N,steps))
model_df.to_csv("{}Agents_{}StepsScipyV6model.csv".format(N,steps))