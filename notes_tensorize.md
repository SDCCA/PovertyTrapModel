## Tensorizing Poverty Trap model ##

Experimental setup:
There are some global fixed variables
including a 'tech' table with gamma (income factor) and cost for that tech

Each agent has:
k: current capital
lambda: saving propensity \in [0.1,1)
alpha: random normal {\mu = 1.08, \sigma = 0.074} "human capital"
sigma: "risk verseness" = 1.5
tec: current tech level
income: current income;
  income generated = alpha * pow(k, tech.gamma) - tech.cost
fronts: tech-income table (not used)
slope: isocline slope (derivative)
consum: current consumption (from isocline - random noise \in [0,1])

Agent step:
  Update capital
    Calculate mean income
    Calculate new capital, consumption, slope
    Calculate conditional consumption and possibly subtract some random amount
    Choose either calculated consumption or conditional consumption
  Give money
    Select a random neighbor of the current agent
    With a probability equal to the weight of the edge with that neighbor
      give a random amount of one of these neighbors' income to their neighbor
      update the edge weights with all the neighbors of the current agent
      update the edge weights with all the neighbors of the neighbor
  Local Attachment (v2)
    Choose a weighted random network edge
    node 1 and 2 are incident to that edge
    From the neighbors of node 2:
      Choose a weighted random neigbor edge
      node 3 is the other node incident to that edge
      If node 1 and 3 do not have an edge, with some fixed probability
        Connect nodes 1 and 3
  Link Deletion
    With some fixed probability:
      Select two random nodes (retry 5 times if these are not connected)
      Remove their edge
  Generate income
    Update the agent's fronts per tec:
      front[i] = alpha * pow(k, tech.gamma) - tech.cost
    Set tec and income based on the front that maximizes income

Model step:
  Perform agent step schedule
  Global Attachment
    With some fixed probability:
      Select two random nodes (retry until these are not connected)
      Connect these nodes
  Compute 'gini' (just for data collector)




