from mesa import Agent, Model
from mesa.time import RandomActivation
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



Version 4 notes:
FinalModelScipyV4 is developed from FinalModelScipyV3.py
In this version, a environmentally responsive theta is introduced. 



Results: 

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

AdapTable = {#contains values for 0:theta 1:cost
# VMG untested and currently unused
    "none":   [0.6, 0   ],
    "good":   [0.8, 0.25],
    "better": [0.9, 0.45]}

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

# def income_function(k, α, γ, tec_cost):

#     return α * k**γ - tec_cost

def income_function(k,α): 
    #print("Generating income")
    f = []
    for i in TechTable.keys(): #in the end, they may each need their own tech table
        entry = α * k**TechTable[i][0] - TechTable[i][1]
        f.append(entry)
    return max(f)
        
def calculate_next_k_and_c(k,c,θ,𝛿,σ,α):
    #print(f"Old k and c {k,c}")
    k_tplus1 = θ * (k - c + (1-𝛿) * k)
    c_tplus1 = solve_bellman(BellmanEquation(u=utility, f=income_function, k=k_tplus1, θ=θ, σ=σ, α=α))
    #print(f"New k and c {k_tplus1,c_tplus1}\n")
    return k_tplus1, c_tplus1

class BellmanEquationStochastic:
    #Adapted from: https://python.quantecon.org/optgrowth.html
    def __init__(self,
                 u,            # utility function
                 f,            # production function
                 k,            # current state k_t
                 σ,            # risk averseness
                 α,            # human capital
                 μ=0.8,        # shock location parameter,for stochastic θ
                 s=0.1,        # shock scale parameter,for stochastic θ
                 β=β,          # discount factor
                 𝛿=𝛿,          # depreciation factor
                 grid_max=10,
                 grid_size=100,
                 shock_size=250,    # shock sample size
                 seed=1234):

        self.u, self.f, self.k, self.β, self.μ, self.s, self.𝛿, self.σ, self.α  = u, f, k, β, μ, s, 𝛿, σ, α

        # Set up grid

        self.grid=np.append(np.array([0.001, 0.01]),np.linspace(0.1, grid_max, grid_size))

        # Identify target state k
        self.index = np.searchsorted(self.grid, k)-1
       


        # Store shocks (with a seed, so results are reproducible)
        self.shocks = np.random.normal(loc=μ, scale=s,size=shock_size)

    def value(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, shocks, 𝛿, σ, α = self.u, self.f, self.β, self.shocks, self.𝛿, self.σ, self.α

        v = interp1d(self.grid, v_array, bounds_error=False, fill_value="extrapolate")
        
        return u(c,σ) + β * np.mean(v((f(y,α) - c + (1 - 𝛿) * y) * shocks))

class BellmanEquation:
     #Adapted from: https://python.quantecon.org/optgrowth.html
    def __init__(self,
                 u,            # utility function
                 f,            # production function, I think the piecewise function may need to be reestablished here for accuracy, more thought needed
                 k,            # current state k_t
                 θ,            # given shock factor θ
                 σ,            # risk averseness
                 α,            # human capital
                 β=β,          # discount factor
                 𝛿=𝛿,          # depreciation factor 
                 grid_max=10,
                 grid_size=100,
                 name="BellmanNarrowExtended"):

        self.u, self.f, self.k, self.β, self.θ, self.𝛿, self.σ, self.α, self.name = u, f, k, β, θ, 𝛿, σ, α, name

        # Set up grid
        
        startgrid=np.array([1.0e-7,1,2,3,4,5,6,7,8,9,10,100])
        #startgrid=np.array([0.001,2,4,6,8,10])

        ind=np.searchsorted(startgrid, k)
        self.grid=np.concatenate((startgrid[:ind],np.array([k-1.0e-06, k]),startgrid[ind:]))
        #self.grid=np.concatenate((startgrid,np.array([k-0.1, k-1.0e-02, k, k+1.0e-02, k+0.1])))

        # Identify target state k
        self.index = np.searchsorted(self.grid, k)-1
    
    def value(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, θ, 𝛿, σ, α = self.u, self.f, self.β, self.θ, self.𝛿, self.σ, self.α

        v = interp1d(self.grid, v_array, bounds_error=False, fill_value="extrapolate")

        return u(c,σ) + β * v(θ*(f(y,α) - c + (1 - 𝛿) * y))


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
        if y <= 1e-8:
            print(f"struggling with {bell.grid} for y={y} for k = {bell.k} and theta={bell.θ}")
        # Maximize RHS of Bellman equation at state y
        c_star, v_max = maximize(bell.value, 1e-8, y, (y, v))
        v_new[i] = v_max
        v_greedy[i] = c_star

    return v_greedy, v_new

def solve_bellman(bell,
                  tol=1,
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

    while i < max_iter and error > tol:
        v_greedy, v_new = update_bellman(v, bell)
        error = np.max(np.abs(v - v_new))
        i += 1
        # if verbose and i % print_skip == 0:
        #     print(f"Error at iteration {i} is {error}.")
        v = v_new

    if error > tol:
        print(f"{bell.name} failed to converge for k={bell.k} with the looser tolerance!")
    elif verbose:
        print(f"Converged in {i} iterations.")
        print(f"Effective k and new c {np.around(bell.grid[bell.index],3),v_greedy[bell.index]}")
        

    return v_greedy[bell.index]







class MoneyAgent(Agent):
    
    def __init__(self, unique_id, model):
        
        super().__init__(unique_id, model)
        self.k = (capital[unique_id]) #initial stock of wealth
        self.lamda = round(random.uniform(0.1,0.949),1)  #VMG replaced lambda <1 while loop
        self.α = alpha[unique_id]#human capital 
        self.σ = round(random.uniform(1,1.9),1)#risk averseness
        self.θ = random.uniform(0.1,1) #percieved theta (cannot be zero or the optimizer breaks because k<)
        self.sensitivity = random.uniform(0,1) #factor controlling tractability of an agent's perception of theta 
        self.tec = "NA"
        self.γ = "NA"
        self.tec_cost = "NA"
        self.income = 0 #initialising income
        self.income_generation() #finding income corresponding to the human capital,
                                 #needed here to set the initial consumption
        self.fronts = 0 #for resetting micawber frontier(s?) 
        self.consum = solve_bellman(BellmanEquation(u=utility, f=income_function, k=self.k, θ=self.θ, σ=self.σ, α=self.α))

        self.model.agents.append(self)

        


      

    #function that decides income based on the type of technology
    def income_generation(self): 
        #print("Generating income")
        f = []
        for i in TechTable.keys(): #in the end, they may each need their own tech table
            entry = self.α * self.k**TechTable[i][0] - TechTable[i][1]
            f.append(entry)
    
        

        #VMG a technology is chosen based on maximizing income
        income = max(f)
        tec = f.index(income)

        γ = TechTableArray[tec][0]
        tec_cost = TechTableArray[tec][1]

        return income,tec,γ,tec_cost
        
            
    
    #function that updates the capital and consumption for the next time step    
    def update_income(self):
        print(f"Updating income and consumption of agent {self.unique_id}")
        self.k, self.consum = calculate_next_k_and_c(self.k,self.consum,self.θ,𝛿,self.σ,self.α)
        
    
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
                    
    def step(self):
        #if(self.k > 0):
        self.update_income()
        self.update_theta()
        self.give_money()
        #self.LocalAttachment_v1()
        self.LocalAttachment_v2()
        self.Link_Deletion()
        self.income_generation() 
        
        
        
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
        
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters = {"Gini": 'gini'},agent_reporters={"k_t":'k','income':'income',
                                           'Fronts':'fronts', 'consumption':'consum','lamda':'lamda','alpha':'α', 'technology':'tec' })       
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
    
    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in tqdm(range(n)):
            #print("Step:", i+1)
            self.time = i+1
            self.step()
            self.Global_Attachment()
            self.gini = self.compute_gini()
            
            
N =100
steps = 50
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
agent_df.to_csv("{}Agents_{}StepsScipyV3.csv".format(N,steps))
model_df.to_csv("{}Agents_{}StepsScipyV3.csv".format(N,steps))