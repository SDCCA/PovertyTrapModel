# From agent data calculates maximum consecutive number and total number of time steps
# spent below a set threshold for each agent at t_final

import pandas as pd


file="V2_500Agents_125Steps.csv"
data=pd.read_csv(file,header=0)
threshold=1

data["InPoverty"]=data["k_t"] < threshold

def max_consec(values):
# Compare poverty bool of step and previous step; if both are true, raise the consecutive day tally. 
# If the running tally is greater than the current maximum, set a new maximum.
# If there were no consecutive days, return 0, otherwise return the maximum (plus one day).

    values.set_index("Step")
    tally,maximum=0,0
    for i in range(1,len(values)):
        if(values.iloc[i]["InPoverty"] == True) & (values.iloc[i-1]["InPoverty"] == True):
            tally+=1
            if tally > maximum:
                maximum=tally
        else:
            tally=0
    if maximum > 0:
        return pd.Series({"MaxConsec":maximum+1})
    return pd.Series({"MaxConsec":0})



def tally_switches(values):
# Compare technology of step and previous step; if equal, pass to the next step. 
# If unequal, add 1 to the Low to High tally if the previous value was "L" or 
# 1 to the High to Low tally if the previous value was not "L". Return the tallies.
    values.set_index("Step")
    LtoHtally,HtoLtally=0,0
    for i in range(1,len(values)):
        if values.iloc[i]["technology"]==values.iloc[i-1]["technology"]:
            pass
        elif values.iloc[i-1]["technology"]=="L":
            LtoHtally+=1
        else:
            HtoLtally+=1
    
    return pd.Series({"LtoH":LtoHtally,"HtoL":HtoLtally})




totalsteps=data.groupby("AgentID").sum("InPoverty")[["InPoverty"]]

consecutive=data.groupby("AgentID").apply(max_consec)

# visually checking differences for feasibility

#print(totalsteps["poverty"].compare(consecutive["consecutive"]))



switches=data.groupby("AgentID").apply(tally_switches)



results=totalsteps.join(consecutive, on="AgentID").join(switches, on="AgentID")

print(results.head())