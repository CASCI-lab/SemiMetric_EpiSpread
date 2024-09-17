import networkx as nx
import numpy as np
import distanceclosure as dc
from distanceclosure.dijkstra import all_pairs_dijkstra_path_length

def create_synthetic_network(G_metric, G_closure, tau, mu, sigma):
    '''
    Create a synthetic network with tunable size of the metric backbone and features of the semi-metric distortion distribution
    
    Parameters
    ----------
    G_metric : NetworkX graph
        The metric backbone of a network. It should be a connected graph and all the edges should constitute the shortest path between their ends.

    G_closure : NetworkX graph
        The metric closure of a network. A graph containing the length of the shortest path connecting every single pair of nodes in the network.

    tau : float
        The relative size of the metric backbone of the synthetic network to be created. It should be between 0 and 1.

    mu, sigma : float
        Parameters of the semi-metric edge distortion log-normal distribution.


    Returns
    ----------
    NetworkX graph
        The synthetic network

    '''

    newG=G_metric.copy()
    G_closure.remove_edges_from(G_metric.edges())
    G_closure_semi_metric=G_closure.copy() 

    E = int((1./tau - 1.)*G_metric.number_of_edges()) 
    semi_edges=list(G_closure_semi_metric.edges()) 
    np.random.shuffle(semi_edges) 
    semi_edges=semi_edges[:E] 

    s_values = np.random.lognormal(mean=mu, sigma=sigma, size=len(semi_edges)) 
    s_values=s_values+1 #### Semi-metric distortion of semi-metric edges should be above 1
    
    for idx, (u,v) in enumerate(semi_edges): 
        d = s_values[idx]*G_closure_semi_metric[u][v]['metric_distance']
        p = 1./(d+1.) 
        newG.add_edge(u,v,**{'distance':d, 'proximity':p,'s_value':s_values[idx],'metric':False,'metric_distance':G_closure_semi_metric[u][v]['metric_distance']})
    
    return newG


def semi_metric_topology(gdf):
    '''
    Computes the metric backbone and semi-metric edge distortion of a given network.

    Parameters
    ----------
    gdf : Pandas Dataframe
        The network dataframe should have two columns, 'source' and 'target' to encode the endpoints of each edge.
        It should also have two columns indicating the "distance" and "proximity" value of the edge.

    Returns
    ----------
    Pandas Dataframe
        Edgelist containing edge distance, proximity, metric_distance, and semi_metric distortion

    '''

    G = nx.from_pandas_edgelist(gdf, source='source', target='target', edge_attr=['distance', 'proximity'])

    # Remove self-loops (just in case)
    print('Remove self-loops')
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    B, svals = dc.metric_backbone(G, weight='distance', distortion=True)
    
    nx.set_edge_attributes(G, name='metric', values=False)
    nx.set_edge_attributes(G, name='metric_distance', values=0.0)

    for (u, v), s in svals.items():
        #d = nx.shortest_path_length(B, source=u, target=v, weight='distance')
        G[u][v]['metric_distance'] = G[u][v]['distance']/s

    for u, v in B.edges():
        svals[(u, v)] = 1.0
        G[u][v]['metric'] = True
        G[u][v]['metric_distance'] = B[u][v]['distance']

    nx.set_edge_attributes(G, name='s_value', values=svals)

    return nx.to_pandas_edgelist(G)

            


def _old_compute_backbone(name,df): ### Function to compute the backbone of a given network introduced as pandas DataFrame
    ### the network dataframe should have two columns, 'source' and 'target' to encode the endpoints of each edge

    # Only keep edges originally in the graph
    df = df.loc[df['original'] == True, :]

    # To NetworkX
    edge_attr = [
        #'b_ij_value',
        #'b_ji_value',
        #'count',
        'distance',
        #'distance_metric_closure',
        #'distance_ultrametric_closure',
        #'metric',
        'original',
        #'proximity',
        #'s_value',
        #'ultrametric',
        'proximity',
        ]
    
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['distance', 'proximity'])

    # Remove self-loops (just in case)
    print('Remove self-loops')
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    print('--- Computing Dijkstra APSP ---')
    #
    # Metric computation
    #
    c = 1
    for i, metric_distances in all_pairs_dijkstra_path_length(G, weight='distance', disjunction=sum):
        if c % 10 == 0:
            print('> Metric Dijkstra: {c:} of {total:}'.format(c=c, total=G.number_of_nodes()))
        for j, metric_distance in metric_distances.items():

            # New Edge?
            if not G.has_edge(i, j):
                # Self-loops have proximity 1, non-existent have 0
                """
                proximity = 1.0 if i == j else 0.0
                G.add_edge(i, j, distance=np.inf, proximity=proximity, metric_distance=float(cm), is_metric=False)
                """
            else:
                G[i][j]['metric_distance'] = metric_distance ## metric distance measures the shortest path distance between the two nodes
                G[i][j]['metric'] = True if ((metric_distance == G[i][j]['distance']) and (metric_distance != np.inf)) else False ## True if the edge belongs to the metric backbone
        c += 1

    print('--- Calculating S Values ---')

    S_dict = {
        (i, j): float(d['distance'] / d['metric_distance']) ### compute distortion value of the edge
        for i, j, d in G.edges(data=True)
        if ((d.get('distance') < np.inf) and (d.get('metric_distance') > 0))
    }
    nx.set_edge_attributes(G, name='s_value', values=S_dict)

   # print('--- Exporting Formats ---')
    #utils.ensurePathExists(wGgraphml)
    #utils.ensurePathExists(wGpickle)
    #utils.ensurePathExists(wGcsv)
    dfG = nx.to_pandas_edgelist(G)

    return dfG ### return dataframe with the network and additional information regarding whether edges are metric or not and their associated semi-metric distortionreturn UG