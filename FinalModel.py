from mesa import Agent, Model
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm 
import seaborn as sns
import networkx as nx
import os#, sys
from itertools import product
import math
#import statistics
#import cmath
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import basinhopping
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import datetime
import atexit



'''VMG notes: 

Version 10 notes:
FinalModelScipyV10 is developed from FinalModelScipyV9.py
In this version, some unused elements have been removed.
This version also accepts variables from the command line (with default
being values used in earlier versions).

Version 10 Outline:
Read in arguments.
Randomly select agent attributes (or import specific agents) and 
 dynamic model parameters.
Write arguments and initial agent data to csv files.
Loop through initialization agent by agent: 
    Assign attributes k #initial stock of wealth, λ-savings propensity,
     α-human capital, σ-risk averseness, and θ-perception of shock.
    (currently no trading at step 0)
    Record generation function technology and resulting income.
    Calculate most advantageous consumption and investment in 
     adaptation using Bellman equation optimization.
DataCollector collects initial step.
Loop through each model step:
    Loop through each agent:
        Update capital based on consumption and adaptation investment 
         optimized in previous time step.
    Loop through each agent giving the option to trade with any 
     neighbors (one opportunity per neighbor pair).
    Loop through each agent:
        Record generation function technology and resulting income.
        Update percieved theta to include observation of global (true) 
         theta at current time.
        Calculate most advantageous consumption and investment in 
         adaptation using Bellman equation optimization.
        Any agent has an opportunity for local attachment or link 
         deletion.
    Global attachment is an opportunity for a single attachment to form
     a randomly selected pair of unconnected agents.
    The list of neighbor pairs is refreshed.
    DataCollector collects step data.
Write relevant model and agent data to csv files.







'''
starttime  = datetime.datetime.now()
# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--no-trading", dest="trading" ,action="store_false", 
                    help="Turn off trading functionality; by default\
                         trading is on.")
parser.add_argument("-s", "--seed", default=42, type=int, 
                    help="Seed for random and np.random generation")
parser.add_argument("-d", "--depreciation", default=0.08, type=float, 
                    help="Depreciation for calculation of k_t+1,\
                         default 0.08")
parser.add_argument("-af", "--agentfile", default=False,type=str, 
                    help="Option to supply csv file (filename.csv) with\
                         custom agents. Column titles must be 'AgentID',\
                            'Alpha','k_0','Sigma','Lambda'. Number of\
                                 agents will be detected automatically.")
parser.add_argument("-ts", "--timesteps", default=100,type=int, 
                    help="Number of time steps, default 100")
parser.add_argument("-fs", "--filesuffix",default=starttime.strftime('%Y-%m-%d_%H_%M_%S') , type=str, 
                    help="A specific suffix can be specified as part of\
                         the output filenames. To include a timestamp, use\
                             \"{args['starttime']}\" in the suffix.")
parser.add_argument("-n", "--agentN",default=500, type=int, 
                    help="Number of agents. Not necessary to specify N if\
                         using file of custom agents.")
parser.add_argument("-a", "--homalpha",default=0.69, type=float, 
                    help="Alpha used in homophily calculation. Not to be confused with agent property alpha")
parser.add_argument("-b", "--homb",default=35, type=float, 
                    help="Characteristic distance used in homophily calculation.")

args = vars(parser.parse_args())
print(args)
trading=args['trading']
seed = args['seed']
random.seed(seed)
np.random.seed(seed)

if not os.path.exists("Results"):
    os.makedirs("Results")

g = 1
p_ga = random.random()#for global attachment
p_ld = random.random()#for link deletion
p_delta = 0.3#for local attachment

g_theta_list = [0.1, 0.5, 0.8, 1]
g_theta_distribution = [0.01 ,0.1, 0.79, 0.1]
expectation = (np.array(g_theta_list)*np.array(g_theta_distribution)).sum()

# global variables for Bellman equation
𝛿 = args["depreciation"] #depreciation
β = 0.95 #discount factor

#Debug
problemAgents=[]#range(500)


TechTable = {#contains values for 0:gamma 1:cost 2:theta
# VMG things get stuck in a while loop if the gamma is less than 0.3 
# (tried 0.2) not sure yet if/how this will be problematic 
# Also important to make sure values are in the correct order
# i.e. that the threshold between medium and high is at a 
# higher k than the threshold between low and medium 
# This can be checked with k_threshold.py

    "low":   [0.3,  0   ],
    "medium":[0.35, 0.15],
    "high":  [0.45, 0.65]}

TechTableArray = np.array([[ 0.3,  0 ],[0.35, 0.15],[0.45, 0.65]])

AdapTable = {
    # contains values for 0:theta 1:cost 
    # (for consideration:effort? type? design life?)
    "none":   [  0, 0   ],
    "good":   [0.5, 0.25],
    "better": [0.9, 0.45]}

AdapTableArray = np.array([[ 0,  0 ],[0.5, 0.25],[0.9, 0.45]])


# global function that calculates the weight of the edge, args: 
# the 2 nodes (agent class objects)
# this is the homophily edge weight formula
def Edge_Weight(node1,node2, b, a):
        try:
             weight = 1+math.exp(a*((node1.k-node2.k)-b))
        except OverflowError:
             weight = float('inf')
        return 1/weight  


def maximize(g, a, b, args):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example 
    https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
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
    f = []
    for i in TechTable.keys(): 
        #in the end, they may need their own tech tables
        entry = α * k**TechTable[i][0] - TechTable[i][1]
        f.append(entry)
    return max(f)

def adaptation_function(θ,i_a):
    
    for i in AdapTable.keys(): 
        #in the end, they should have their own adaptation tables
        if AdapTable[i][1] <= i_a:
            m = AdapTable[i][0]
        else:
            break
    return θ + m * (1-θ)

        

        
def calculate_next_k(agentinfo):
    #formula for k_t+1 is applied at the beginning of each time step 
    # k_t+1 becomes the new k_t
    
    k,c,i_a,m,α = agentinfo.k,agentinfo.consum,agentinfo.i_a,agentinfo.m,agentinfo.α
    if agentinfo.unique_id in problemAgents:
        print(f"Calculating using k={k}, global_theta={global_θ[model.time]}, m={m}, c={c}, i_a={i_a}, sigma={agentinfo.σ}, alpha={agentinfo.α}, percieved theta={agentinfo.θ}")
    k_tplus1 = (global_θ[model.time] + m * (1-global_θ[model.time])) * (income_function(k,α) - c - i_a + (1-𝛿) * k)
    
    if agentinfo.unique_id in problemAgents:
   
        print(f"k_t after Calculate={k_tplus1}")
    return k_tplus1



class BellmanEquation:
     #Adapted from: https://python.quantecon.org/optgrowth.html
    def __init__(self,
                 u,            # utility function
                 f,            # production function
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
        
        startgrid=np.array([1.0e-7,1,2,3,4,5,6,7,8,9,10,k+100])

        ind=np.searchsorted(startgrid, k)
        self.grid=np.concatenate((startgrid[:ind],np.array([k*0.99999, k]),
                                 startgrid[ind:]))

        self.grid=self.grid[self.grid>i_a]

        # Identify target state k
        self.index = np.searchsorted(self.grid, k)-1
    
    def value(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, θ, 𝛿, σ, α, i_a, m = self.u, self.f, self.β, self.θ, self.𝛿, self.σ, self.α, self.i_a, self.m

        v = interp1d(self.grid, v_array, bounds_error=False, 
                     fill_value="extrapolate")
        
        return u(c,σ) + β * v((θ + m * (1-θ)) * (f(y,α) - c - i_a + (1 - 𝛿) * y))



def update_bellman(v, bell):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example
    https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    
    The Bellman operator.  Updates the guess of the value function
    and also computes a v-greedy policy.

      * bell is an instance of Bellman equation
      * v is an array representing a guess of the value function

    """
    v_new = np.empty_like(v)
    v_greedy = np.empty_like(v)
    
    for i in range(len(bell.grid)):
        y = bell.grid[i]
        # Maximize RHS of Bellman equation at state y
        
        c_star, v_max = maximize(bell.value, min([1e-8,y*0.00001]), 
                                 y-bell.i_a, (y, v))
        #VMG HELP! can anyone check that (1) subtracting i_a and 
        # (2) omitting any grid values less than i_a 
        # will not be problematic? The only thing I can come up with
        # is if i_a is greater than k*0.99999
        # which_bellman() now accounts for that case. Whole thing 
        # could use refinement.
      
        v_new[i] = v_max
        v_greedy[i] = c_star

    return v_greedy, v_new

def which_bellman(agentinfo):
    """
    Solves bellman for each affordable adaptation option.
    """
    feasible=[]
    if agentinfo.unique_id in problemAgents:
        print(f" k= {agentinfo.k} passed to which_bellman")

    for option in agentinfo.adapt:
        if option[1]>=agentinfo.k*0.99998:
            # ensures that the gridpoint
            # just below k, k*0.99999, is included
            pass
        else:
            #print(f'working theta = {agentinfo.θ + option[0] *\
            #  (1-agentinfo.θ)}, i_a= {option[1]}, k= {agentinfo.k}')
            c,v=solve_bellman(BellmanEquation(u=utility, 
                              f=income_function, k=agentinfo.k, 
                              θ=agentinfo.θ, σ=agentinfo.σ, 
                              α=agentinfo.α, i_a=option[1],m=option[0]))
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
    From: https://python.quantecon.org/optgrowth.html (similar example
    https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    
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
        self.k = capital[unique_id] #initial stock of wealth
        self.λ = lambdas[unique_id] #savings propensity
        self.α = alphas[unique_id]#human capital 
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
        self.connections=0 #tracks number of neighbors 
        self.trades=0 #tracks number of trades per timestep
        self.net_traded=0 #tracks net gain/loss from trades
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

        # a technology is chosen based on maximizing income
        self.income = max(f)
        self.tec = f.index(self.income) 

        
    def initialize_consumption(self):
        #print(f'\nInitializing agent {self.unique_id}')
        self.consum, self.i_a, self.m =which_bellman(self)
        
    
    def update_capital(self):    # updates the capital, consumption, investment, and theta multiplier for the next time step    
        if self.unique_id in problemAgents:
            print(f"\nUpdating capital of agent {self.unique_id} at time step {model.time}")
        self.k = calculate_next_k(self)
        self.connections=0
        self.trades=0
        self.net_traded=self.k

    def update_consumption(self):
        if self.unique_id in problemAgents:
            print(f"\nUpdating consumption of agent {self.unique_id} at time step {model.time}")
            print(f"current k ={self.k} k after capital update was {self.net_traded}")
        self.net_traded=self.k-self.net_traded
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
        if self.unique_id in problemAgents:
            print(f"{self.unique_id} is starting to trade")
        b = self.model.b
        a = self.model.a
        neighbor_IDs = []
        for pair in self.model.trade_partners:
            if self.unique_id in pair:
                neighbor=list(pair)
                neighbor.remove(self.unique_id)
                if len(neighbor)>0:
                    neighbor_IDs.append(neighbor[0])
        neighbors=[]
        for agent in self.model.agents:
            if(agent.unique_id in neighbor_IDs):
                neighbors.append(agent)
        epsilon = random.random()
        for i in neighbors:
            other = i
            if(other.unique_id == self.unique_id):
                continue
            self.connections+=1
            other.connections+=1  
            w = self.model.G[self.unique_id][other.unique_id]['weight']
            if self.unique_id in problemAgents:
                    print(f"Weight of edge is {w} vs {model.min_weight}")
            delta_money=0 
            if(w >= model.min_weight):
                self.trades+=1 
                other.trades+=1 
                xi = self.k
                xj = other.k
                #self.money_traded = epsilon * ((1-self.λ) * self.k + (1-other.λ) * other.k)
                delta_money = (1-self.λ) * self.k - epsilon * ((1-self.λ) * self.k + (1-other.λ) * other.k)
                self.k = xi - delta_money
                other.k = xj + delta_money
                if self.unique_id in problemAgents:
                    print(f"Removing from trade partners {self.unique_id,other.unique_id}")
                self.model.trade_partners.remove({self.unique_id,other.unique_id})
                for neighbor in neighbors:
                    self.model.G[self.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(self,neighbor,b, a)
                other_neighbors = other.neighbors()
                for neighbor in other_neighbors:
                    if(neighbor.unique_id != other.unique_id):
                        self.model.G[other.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(other,neighbor,b, a)
            if self.unique_id in problemAgents:
                print(f"Trade pool between agent {self.unique_id} and agent {other.unique_id} for timestep {model.time} is {delta_money}")
                print(f"self k = {self.k}   other k = {other.k}")

    def update_theta(self):#conducted by agent at each time step, the observed theta impacts agent perception of theta
        self.θ=self.θ * (1-self.sensitivity) + global_θ[model.time] * self.sensitivity

        
        
    #1. select nodes i and j, with a probability proportional to the weight (wij) between them.
    #2. j selects a neighbour with a probability proportional to the weight between them such that there is no edge between 
    #i and k. 
    #3. a link is made between i and k with edge weight w0 (here, w0 = 1) and all edge weights are increased by wr(wr-calculated 
    #new edge weight between i and j)
    def LocalAttachment_v2(self):
        b = self.model.b
        a = self.model.a
        links_nodes = list(self.model.G.edges(data= 'weight'))
        edge_weight_sum = sum(k for i, j,k in links_nodes)
        if edge_weight_sum==0:
            print(links_nodes)

        edge_weights = []
        for edge in links_nodes:
            edge_weights.append(edge[2])
        edge_prob = []
        for edge_w in edge_weights:
            edge_prob.append(edge_w/edge_weight_sum)
        arr_pos = [i for i in range(len(links_nodes))]
        pos = np.random.choice(arr_pos, p = edge_prob)
        chosen = links_nodes[pos]
        node1 = chosen[0]
        node2 = chosen[1]
        node1_a = next((agent for agent in self.model.agents if agent.unique_id == node1), None)
        node2_a = next((agent for agent in self.model.agents if agent.unique_id == node2), None)

        neighbors = [n for n in self.model.G.neighbors(node2)]
        if(len(neighbors)>1):
            links_nodes = list(self.model.G.edges(node2, data= 'weight'))
            edge_weight_sum = sum(k for i, j,k in links_nodes)
            edge_weights = []
            for edge in links_nodes:
                edge_weights.append(edge[2])
            edge_prob = []
            for edge_w in edge_weights:
                edge_prob.append(edge_w/edge_weight_sum)
            arr_pos = [i for i in range(len(links_nodes))]
            pos = np.random.choice(arr_pos, p = edge_prob)
            chosen = links_nodes[pos]
            for node in chosen[0:2]: #because the 3rd element is weight
                if(node!= node2):
                    node3 = node
                    if(self.model.G.has_edge(node1,node3)==False and random.random()>p_delta):
                        node3_a = next((agent for agent in self.model.agents if agent.unique_id == node3), None)
                        self.model.G.add_edge(node1,node3,weight = Edge_Weight(node1_a,node3_a, b, a))
                    
    
   #links are deleted randomly at every time step
    def Link_Deletion(self):
        if(random.random()>p_ld):
            node1 = random.choice(list(self.model.nodes))
            node2 = random.choice(list(self.model.nodes))
            count = 0
            while(self.model.G.has_edge(node1,node2)==False and count<5):
                node2 = random.choice(list(self.model.nodes))
                count +=1
            if(count !=5):
                self.model.G.remove_edge(node1,node2)
                    
    def stageA(self):
        self.update_capital()

    def stageB(self):
        if trading==True:
            self.trade_money()
        if trading==False:
            pass

    def stageC(self):
        self.record_income()
        self.update_theta()
        self.update_consumption()
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
        self.trade_partners=[]   
        self.Refresh_Trade()
        stage_list=["stageA","stageB","stageC"]
        self.schedule = StagedActivation(self,stage_list,shuffle=True)
        self.datacollector = DataCollector(model_reporters = {"Gini": 'gini', "Global_Theta":'globe_theta',"Trade_Threshold":'min_weight'},agent_reporters={"k_t":'k','income':'income',
                                           'Fronts':'fronts', 'consumption':'consum','lambda':'λ','alpha':'α', 'percieved_theta':'θ', 'technology':'tec', "i_a":"i_a","connections":"connections","trades":"trades", "net_traded":"net_traded"})       
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

    def Refresh_Trade(self):
        #Creates a list of unique connections
        self.trade_partners=[]
        for agent in self.G.nodes:
            neighbor_nodes = list(nx.all_neighbors(self.G,agent))
            for other in neighbor_nodes:
                entry={agent,other}
                if entry not in self.trade_partners:
                    self.trade_partners.append(entry)

    def compute_gini(self):
        agent_wealths = [agent.k for agent in self.schedule.agents]
        x = sorted(agent_wealths)
        B = sum(xi * (self.N - i) for i, xi in enumerate(x)) / (self.N * sum(x))
        return 1 + (1 / self.N) - 2 * B

    def write_data(self):
        model_df = self.datacollector.get_model_vars_dataframe()
        agent_df = self.datacollector.get_agent_vars_dataframe()
        agent_df.reset_index(level=1, inplace = True)
        agent_df.to_csv(f"Results/ScipyV10_{args['filesuffix']}.csv")
        model_df.to_csv(f"Results/ScipyV10_{args['filesuffix']}model.csv")
        endtime=datetime.datetime.now()
        with open(f"Results/ScipyV10_{args['filesuffix']}params.txt", 'a') as pfile:
            pfile.write(f"Start Time: {starttime.strftime('%Y-%m-%d %H:%M:%S')}  End Time: {endtime.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed Time: {endtime-starttime}")
        print(f"Exiting at model time {model.time}")
        
    
    def step(self):#Model level step
        # staged activation
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, steps):
        for i in tqdm(range(1,steps),ascii=True,desc="Progress"):
            self.time = i
            self.globe_theta = global_θ[self.time]
            self.min_weight = min_weights[self.time]
            self.step()
            self.Global_Attachment()
            self.Refresh_Trade()
            self.gini = self.compute_gini()
        atexit.register(self.write_data)
            
            
            

steps = args["timesteps"]
b = args["homalpha"]
a = args["homb"]
if args["agentfile"]!=False:
    custom_agents=pd.read_csv(args["agentfile"],header=0,usecols=["AgentID","Alpha","k_0","Sigma","Lambda"],index_col=0)
    N = len(custom_agents)
    alphas = custom_agents["Alpha"] #list of agent alphas, effectively ability/human capital
    capital = custom_agents["k_0"] #list of agent initial capital amounts
    sigmas= custom_agents["Sigma"] #list of agent sigmas, effectively risk-averseness
    lambdas= custom_agents["Lambda"] #list of agent lambdas, effectively savings propensity

else:
    N = args["agentN"]
    alphas = np.random.normal(loc = 1.08, scale = 0.074, size = N) #list of agent alphas, effectively ability/human capital
    capital = np.random.uniform(low = 0.1, high = 10, size = N) #list of agent initial capital amounts
    sigmas=np.round(np.random.normal(loc = 1, scale = 0.5,size=N),1) #list of agent sigmas, effectively risk-averseness
    lambdas=np.round(np.random.uniform(0.1,0.949,size=N),1) #list of agent lambdas, effectively savings propensity

global_θ = np.random.choice(g_theta_list, steps ,p=g_theta_distribution).tolist() #list of model-level thetas, effectively schedule of system shocks
min_weights = np.random.random(steps) #threshold referenced at each time step for trades between agent pairs
initialdf=pd.DataFrame(data={"AgentID":range(N),"Alpha":alphas,"k_0":capital,"Sigma":sigmas,"Lambda":lambdas}).set_index("AgentID")


if args["agentfile"]!=False: 
    if custom_agents.equals(initialdf):
        print(f"Import of {len(custom_agents)} agents successful.")
    else:
        print("Agent import attempted but not successful. Please see formatting guidelines and reattempt.")

with open(f"Results/ScipyV10_{args['filesuffix']}params.txt",'w') as pfile: 
    pfile.write(str(args)+'\n')
initialdf.to_csv(f"Results/ScipyV10_{args['filesuffix']}properties.csv")

model = BoltzmannWealthModelNetwork(b, a, N)
model.run_model(steps)

