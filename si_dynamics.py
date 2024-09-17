from utils import*
import numpy as np
import random

def initiate_configuration(seed,nodes):
    '''
    Set the infectious seed for the SI dynamics
    '''
    
    states={i:0 for i in nodes}
    states[seed]=1
    
    return states


def update_inf_probabilities(beta, dt, out_neighbors, out_weights, probability_of_infection, new_nodes_infected):
    '''
    Updates the probability of infection of the neighboring nodes of those contracting the disease at a given time step.

    Parameters
    ----------
    beta : float
        Contagion rate.

    dt : float
        Length of the discrete time step considered in the simulaitons

    out_weights : list of list
        Contains the set of weights connecting one individual with its neighbors.
    
    probability_of_infection : list
        Probability that a node gets the disease according to the dynamical state of the set of neighbors.
    
    new_nodes_infected : list
        Containing the nodes contracting the disease at a given time step.

    Returns
    ----------
    list
        The update probability of infection.

    '''

    for i in new_nodes_infected:
        for j in range(len(out_neighbors[i])):
            probability_of_infection[out_neighbors[i][j]]=1.0-(1.0-probability_of_infection[out_neighbors[i][j]])*(1.0-beta*dt*out_weights[i][j])

    return probability_of_infection
    
def simulateSI(seed, beta, dt, network): 
    '''
    Simulate the SI dynamics considering a single seed of infection in a given network (DataFrame format)
    '''

    #Initiate the dynamics
    nodes=node_list(network) 
    size=len(nodes)
    states=initiate_configuration(seed,nodes) #Initial configuration of the individuals
    out_neighbors,out_weights=get_neighborhood_properties(network) #Get list of neighbors and weights
    time_of_infection={i:0 for i in nodes}#Dictionary to track the time of infection of each node
    probability_of_infection={i:0 for i in nodes}#Set the probability of infection for each node
    susceptible_nodes=nodes.copy()
    susceptible_nodes.remove(seed) #Identify the susceptible nodes

    #Get the probability of infection of those nodes in contact with the infectious seed
    probability_of_infection=update_inf_probabilities(beta,dt,out_neighbors,out_weights,probability_of_infection,[seed])
    for key,items in probability_of_infection.items():
        if np.isnan(items)==True:
            print('ERROR')

    ninf=1 ### Count the number of infected individuals
    step=0 ## Set the current time step
    while(ninf!=size): ### Iterate SI dynamics until all the nodes get infected
        step+=1  ## Advance one time step
        new_contagions=[i for i in susceptible_nodes if random.random()<probability_of_infection[i]] ### List with the new contagions at this time step
        new_contagions=list(set(new_contagions)) ### A node can get infected multiple times in a time step if it's neighbor of multiple infected individuals. For this reason, we must remove duplicates
        for i in new_contagions:
            time_of_infection[i]=step*dt ## Set the time of infection of nodes getting the disease at the current time step 
            susceptible_nodes.remove(i) ## remove them from the list of susceptible nodes
            ninf+=1 ### updates the number of infected individuals in the population
        probability_of_infection=update_inf_probabilities(beta,dt,out_neighbors,out_weights,probability_of_infection,new_contagions) ### update the probability of infection of the remaining nodes

    return time_of_infection ### return the dictionary with the times of infection of each node


def get_si_results(network_name):
    '''
    Generate a dataframe with the SI simulation for multiple parameters.
    '''

    network_original=pd.read_csv('Data/network_with_semi_metric_topology_%s.csv'%network_name)
    metric_network=network_original[network_original['metric']==True]
    
    realizations=30 ### Number of outbreaks simulated for each infectious seed
    beta=0.5 ### Infectiousness parameter
    dt=0.1 ### Duration of each time step
    nodes=node_list(network_original) #Compute nodes of the network
    results=[]
    seeds=random.choices(nodes,k=10) ### Set k different locations for the seeds. IMPORTANT: When comparing with other methods/sizes of the network, keep the same set of nodes as seeds for a fair comparison
    semi_metric_network=network_original[network_original['metric']==False].sort_values('s_value',ascending=True) ### All semi-metric edges sorted from the lowest to the highest distorsion values
    chi_values = np.arange(0,1.01,0.05) ## chi controls the size of the sparsified network. Fraction of semi-metric edges remaining in the network after sparsifying. chi=1 original network, chi=0 metric backbone
    
    for chi in chi_values: 
        sample_size=int(chi*len(semi_metric_network)) ### Number of semi-metric edges included
        ## Keep those semi-metric edges with lowest distorsion values
        #network_used=metric_network.append(semi_metric_network.head(sample_size)) 
        #### NOTE: UPDATED TO THE NEW PANDAS VERSION
        network_used=pd.concat([metric_network, semi_metric_network.head(sample_size)]) 
        for seed in seeds:
            for real in range(realizations):
                time_distribution=list(simulateSI(seed,beta,dt,network_used).values()) ## get a dictionary with the times of infection of all nodes
                results.append(tuple([beta,chi,network_name,seed,real,[np.quantile(time_distribution,x).round(1) for x in np.arange(0.1,1.01,0.1)]])) ### append for each realization and seed
                real+=1
            print('Chi %.2f, Seed %d is done'%(chi,seed))
    
    # Create a dataframe with the parameters and the result (last column). Last column corresponds to a list of times at which a given quantile of nodes (10%,20%,30%...) gets infected.
    df_results=pd.DataFrame(results,columns=['beta','size','network','seed','realization','times']) 

    return df_results
