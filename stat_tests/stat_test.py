import pandas as pd
import argparse
import seaborn as sns
import stat_test_utils as stu
import os

defaultbaseline ="../test_data/V2_500Agents_125StepsSeed1.csv"
defaultpovertythreshold = 1.

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument( "--simulation","-s",help="path to simulation output to be evaluated for similarity ", type=str, required=True)
	parser.add_argument( "--baseline", "-b", help="Optional. Path to simulation output to be used as baseline", type=str, default=defaultbaseline)
	parser.add_argument( "--povertythreshold", "-pt", help="Value for poverty threshold used", type=float,default=defaultpovertythreshold)

	args=parser.parse_args()
	return args


def load_sim_file(path,*kwargs):
	data=[]
	if os.path.isfile(path):
		try:
			data = pd.read_csv(path,*kwargs)
		except:
		    print(f"failed to read {path}")
		
	else:
	    print(f"{path} is not a file")
	return data									


def print_stat_test_output(pvalues,properties):
    if len(pvalues) != len(properties):
        print('mismatch in properties and p values')
    else:
        for pv, prop in zip(pvalues,properties):
            if pv > 0.05:
                print(f"Null hypothesis of same parent distribution accepted for {prop} at p = {pv} \n")
            else:
                print(f"Null hypothesis of same parent distribution rejected for {prop} at p = {pv} \n")



def main():
    args = parse_args()
    
    print('loading data ...\n')
    sim_data = load_sim_file(args.simulation)
    base_data = load_sim_file(args.baseline)

    """
    calculate derived properties
    (i) poverty 
    """

    print("calculating derived properties ...\n")
    sim_data["InPoverty"] = sim_data["k_t"] < args.povertythreshold
    base_data["InPoverty"] = base_data["k_t"] < args.povertythreshold


    sim_total_steps_in_poverty = sim_data.groupby("AgentID").sum("InPoverty")[["InPoverty"]]
    sim_max_consec_steps = sim_data.groupby("AgentID").apply(stu.max_consec)

    base_total_steps_in_poverty = base_data.groupby("AgentID").sum("InPoverty")[["InPoverty"]]
    base_max_consec_steps = base_data.groupby("AgentID").apply(stu.max_consec)

    """
    (ii) Technology regime switches
    """

    sim_switches=sim_data.groupby("AgentID").apply(stu.tally_switches)

    base_switches=base_data.groupby("AgentID").apply(stu.tally_switches)

    """
    join derived properties for further analysis
    """

    sim_derived = sim_total_steps_in_poverty.join(sim_max_consec_steps, on="AgentID").join(sim_switches, on="AgentID")
    base_derived = base_total_steps_in_poverty.join(base_max_consec_steps, on="AgentID").join(base_switches, on="AgentID")


    """
    Perform two sample statistical distribution tests on final distributions of simulated properties
    """
    print("performing Cramer von Mises wo sample tests ...\n")
    #Cramer von mises
    cvm_kt_p, cvm_kt_s = stu.CramerVonMises(sim_data,base_data,'k_t',collateSteps=True)

    cvm_total_poverty_p, cvm_total_poverty_s = stu.CramerVonMises(sim_derived,base_derived,'InPoverty')
    cvm_consec_poverty_p, cvm_consec_poverty_s = stu.CramerVonMises(sim_derived,base_derived,'MaxConsec')

    cvm_LtoH_p, cvm_LtoH_statistic = stu.CramerVonMises(sim_derived,base_derived,'LtoH')
    cvm_HtoL_p, cvm_HtoL_statistic = stu.CramerVonMises(sim_derived,base_derived,'HtoL')


    """
    Print results.

    TODO: Tie into CI
    """
    print_stat_test_output([cvm_kt_p,cvm_total_poverty_p,cvm_consec_poverty_p,cvm_LtoH_p,cvm_HtoL_p],['k_t','total steps in poverty','max consecutive steps in poverty','switches from L to H technology','switches from H to L technology'])
    





if __name__ == "__main__":
    main()	


