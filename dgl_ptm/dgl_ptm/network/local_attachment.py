#!/usr/bin/env python
# coding: utf-8

# local_attachment - Creates links between agents with "neighbor of a neighbor" approach

import torch
import dgl

def local_attachment(graph,n_FoF_links,edge_prop=None,p_attach=1.):
    created_links = 0
    if edge_prop != None:
        src_ids, dst_ids, _ = select_edges(graph,n_FoF_links,edge_prop)
        for i in range(len(src_ids)):
            FoF_to_link = select_FoF_attachment(src_ids[i],dst_ids[i],graph,edgeprop=edge_prop)
            if FoF_to_link != None:
                create_FoF_link (src_ids(i),FoF_to_link,graph,p_attach=p_attach)
                created_links +=1
            else:
                print(f"no FoF link possible for src dst nodes {src_ids[i]} and {dst_ids[i]}.")
        print(f'created {created_links} of {n_FoF_links} links requested')
    else:
        raise RuntimeError('edge property for local attachement must be specified')

def select_edges(graph,n_FoF_links,edge_prop):
    norm_edge_prop = graph_norm_edge_prop(graph,edge_prop)
    selected_edges=norm_edge_prop.flatten().multinomial(num_samples=n_FoF_links,replacement=False)
    src_ids = graph.edges('all')[0][selected_edges]
    dst_ids = graph.edges('all')[1][selected_edges]
    e_ids = graph.edges('all')[2][selected_edges]
    return src_ids, dst_ids, e_ids


def graph_norm_edge_prop(graph,edgeprop):
    if graph.is_homogenous:
        return graph.edata[edgeprop]/graph.edata[edgeprop].sum()
    else:
        raise RuntimeError('only homogenous graphs are currenlty supported')

def FoF_nodes(srcid,dstid,graph):
    """
    For an edge in a homogenous graph defined by its src and dst nodes find all Friends-of-Friends (FoF), defiined as (downstream) neighbours of the dst node,
    removing src node as may result from bidirectional edges. For a bidirectional graph this is equivalent to all neighbours.
    """
    successors_ = graph.successors(dstid)
    successors = successors_[successors_!=srcid]
    return successors

def existing_connections(srcid,nodelist,graph):
    """
    identify all exisitng links between a src node and a list of dst nodes 
    """
    existing_connection=graph.has_edges_between(srcid,nodelist)
    return existing_connection 

def select_FoF_attachment(srcid,dstid,graph,edgeprop=None):
    """
    select possible FoF attachment by identifying potential FoF nodes {F} of the src-dst edge, discarding exisiting  src-F connections and selecting a
    possible FoF attachment target node T E {F} by weighted draw form the normalized weight distribution of all edges dst-{F}
    """
    selected_FoF=None
    possible_FoF_nodes = FoF_nodes(srcid,dstid,graph)
    if possible_FoF_nodes.numel() !=0:
        already_connected = existing_connections(srcid,possible_FoF_nodes)
        if torch.all(already_connected):
            print(f'all FoF nodes are already irecdctly connected to node {srcid}.')
        else:
            possible_FoF_to_link = possible_FoF_nodes[already_connected==False]
            dst_F_link_ids = graph.edge_ids(torch.ones_like(possible_FoF_to_link,dtype=int)*dstid,possible_FoF_to_link)
            dst_F_link_weight = graph.edges[dst_F_link_ids].data[edgeprop]
            dst_F_link_weight_norm = dst_F_link_weight/dst_F_link_weight.sum()
            select = dst_F_link_weight_norm.flatten().multinomial(num_samples=1,replacement=True)
            selected_FoF = possible_FoF_to_link[select]
    else:
        print(f"dst node {dstid} is an end node")
    return selected_FoF

def create_FoF_link(srcid,targetid,graph,p_attach=1.):
    create_link = False
    if p_attach < 1.:
        if p_attach > torch.rand(1):
            create_link = True
    else:
        create_link = True
    if create_link:
        print(f"creating bidirectional link between nodes {srcid} (src) and {targetid} (dst)")
        graph.add_edges(srcid,targetid)
        graph.add_edges(targetid,srcid)




    