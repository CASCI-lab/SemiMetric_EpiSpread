import networkx as nx
import pandas as pd


def compute_size_network(network):
    '''
    Returns the number of nodes in the network, network is a dataframe where there should be at least two columns with the source and target of each edge    
    '''

    return len(list(set(list(network['source'].unique())+list(network['target'].unique()))))


def node_list(network):
    '''
    Returns the list of nodes in the network, network is a dataframe where there should be at least two columns with the source and target of each edge
    ''' 
 
    return list(set(list(network['source'].unique())+list(network['target'].unique())))


def get_neighborhood_properties(network):
    '''
    This fucntion produces a dictionary with all the neighbors of a node and the corresponding weights of their interactions
    '''

    nodes=node_list(network)
    out_neighbors={i:[] for i in nodes}
    out_weights={i:[] for i in nodes}
    for index, row in network.iterrows():
        if(row['proximity']!=0):
            out_neighbors[row['source']].append(row['target'])
            out_weights[row['source']].append(row['proximity'])
            out_neighbors[row['target']].append(row['source'])
            out_weights[row['target']].append(row['proximity'])

    return out_neighbors,out_weights


def sort_edges(df):
    '''
    Sort edges according to their source and target
    '''  
    
    sources=[]
    target=[]
    for index,row in df.iterrows():
        if (row['source']>row['target']):
            sources.append(row['target'])
            target.append(row['source'])
        else:
            sources.append(row['source'])
            target.append(row['target'])
    df['source']=sources
    df['target']=target
   
    return df


def directed_network_to_undirected(G): 
    '''
    Symmetrize a weighted network by considering the sum of weights
    '''
    
    UG = G.to_undirected()
    for node in G:
        for ngbr in nx.neighbors(G, node):
            if node in nx.neighbors(G, ngbr):
                UG.edges[node, ngbr]['proximity'] = (
                    G.edges[node, ngbr]['proximity'] + G.edges[ngbr, node]['proximity']
            )
    UG.edges.data('proximity')

    return UG


def network_from_raw_data(city):
    '''
    Generates the dataframe from the raw transportation data.
    '''

    #### Read mobility data
    mobilitynetwork=pd.read_csv('Data/mobility_network_%s.csv'%city)
    ## convert to a networkx object
    G = nx.from_pandas_edgelist(mobilitynetwork, source='source', target='target', edge_attr=['proximity'],create_using=nx.DiGraph()) 
    G = directed_network_to_undirected(G) ## We consider a undirected network representation
    G = G.subgraph(max(nx.connected_components(G), key=len)) ## Get the largest connected component
    mobilitydef = nx.to_pandas_edgelist(G) ### dataframe with the clean network
    
    ####Compute Jaccard normalization
    strength_source=mobilitydef.groupby('source')['proximity'].sum().reset_index()
    strength_source.columns=['node','strength_source'] ### Compute strength of all the connection where one node is the source
    strength_target=mobilitydef.groupby('target')['proximity'].sum().reset_index()
    strength_target.columns=['node','strength_target'] ### Compute strength of all the connection where one node is the target
    
    strength_total=pd.merge(strength_source,strength_target,on='node',how='outer') ### Get a dataframe with both strengths
    strength_total=strength_total.fillna(0) ## In case one node is not a target or a source of any connections
    strength_total['strength']=strength_total['strength_source']+strength_total['strength_target'] ### Get the strength of the node
    
    strength_total=strength_total[['node','strength']] 
    strength_total.columns=['source','strength_source'] 
    
    mobilitydef=pd.merge(mobilitydef,strength_total,on='source')
    strength_total.columns=['target','strength_target']
    
    mobilitydef=pd.merge(mobilitydef,strength_total,on='target') ###get dataframe indicating the weight of the edge and the strength of the endpoints
    mobilitydef['proximity']=mobilitydef['proximity']/(mobilitydef['strength_source']+mobilitydef['strength_target']-mobilitydef['proximity']) ### Apply Jaccard normalization to the weights
    mobilitydef['distance']=1.0/mobilitydef['proximity']-1 ## Apply isomorphism to get the associated distance to each edge

    return mobilitydef

