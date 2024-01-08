from PIL import Image, ImageDraw, ImageFont
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
from itertools import combinations
import ipywidgets as wg
from IPython.display import SVG, display
import copy
import gc
from collections import defaultdict

import scipy
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import t, linregress, norm
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, vstack

from collections import defaultdict, deque
import networkx as nx
import numbers
import json
import time
from sklearn import metrics
import joblib
from numba import jit
import math
import pandas as pd
import random
from statsmodels.stats.multitest import multipletests
import threading



try:
    from app.network_data.basic_functions import *
    from app.network_data.edge import *
    from app.network_data.node import *
except:
    from network_data.basic_functions import *
    from network_data.edge import *
    from network_data.node import *
    
class ModelDict(dict):
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__()
        
    def __iter__(self):
        return iter(self.values())
    
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
    
class Model:

    def __init__(self, grid = (1,1), seed = 0):
        self.nodes = ModelDict(self)
        self.edges = ModelDict(self)
        self.entities = ModelDict(self)
        self.grid = grid
        self.grid_size = self.grid[0] * self.grid[1]
        self.files = {}
        self.submap_images = {}
        self.nodes_with_changes = set()
                        
        if not seed:
            seed = random.randint(0, 10000)
        self.seed = seed

    def __hash__(self):
        return hash((self.nodes, self.edges, self.grid))
    
    def get_node_from_name(self, name, compartment = None, nodetype = None, states = None):
        for node in self.nodes:
            if node.name.lower() == name.lower() and (compartment == None or node.compartment.lower() == compartment.lower()) and (states == None or sorted(node.states) == sorted(states)) and (nodetype == None or node.type == nodetype.lower()):
                return node
        return None
    
    def adjust_indices(self):
        for i,node in enumerate(self.nodes):
            node.index = i

    def get_nodes_from_name(self, name, exact = True):
        if exact:
            return [node for node in self.nodes if node.name.lower() == name.lower()]
        else:
            return [node for node in self.nodes if name.lower() in node.name.lower()]         
    
    def get_nodes_from_names(self, names, exact = True):
        return [node for node in sum([self.get_nodes_from_name(name, exact = exact) for name in names], []) if node]
    
    def print_edges(self):
        return [edge.as_string() for edge in self.edges]
    
    def print_nodes(self):
        return [node.as_string() for node in sorted(self.nodes)]

    def get_nodes_by_type(self, node_type):
        return [node for node in self.nodes if node.type.lower() == node_type.lower()]    
    
    @property
    def phenotypes(self):
        return [node for node in self.get_nodes_by_type("phenotype") if not node.hypothetical]
    
    @property
    def compartments(self):
        return list(set([node.compartment for node in self.nodes]))
    
    def update_signaling(self):
        for node in self.nodes:
            node.boolean_targets = []
        for node in self.nodes:
            node.update_rule()
            
    def connect_rna_proteins(self):
        for node1 in self.nodes:
            if node1.type == "rna":
                for node2 in node1.siblings:
                    if node2.type == "protein" and node2.compartment == node1.compartment and len(node2.states) == 0:
                        Edge(
                            self,
                            [node1], 
                            [node2],
                            edgetype = "positive", 
                        )
    
    def write_json_files(self, path):
        elements_json = {}
        interaction_json = []

        print("Generate Elements")
        for node in self.nodes:
            elements_json[node.string_index] = {
                "fullname": node.fullname_printable,
                "minerva_name": node.minerva_name.lower(),
                "name": node.name,
                "type": node.type_class,
                "subtype": node.type,
                "submap": node.submap,
                "hash": node.hash_string,
                "ids": {
                    "name": node.name
                }, 
                "family": [parent.string_index for parent in self.nodes if parent.family and node in parent.subunits],
                "parent": [parent.string_index for parent in self.nodes if not parent.family and node in parent.subunits],
                "subunits": [subunit.string_index for subunit in node.subunits]
            }
            
        print("Generate Interactions")
        for edge in self.edges:
            for edge_json in edge.as_simple_json():
                edge_json["source"] = edge_json["source"].string_index
                edge_json["target"] = edge_json["target"].string_index
                interaction_json.append(edge_json)

        print("Write Elements")
        json_object = json.dumps(elements_json)
        with open(path + "Elements.json", "w") as outfile:
            outfile.write(json_object)
            
        print("Write Interactions")
        json_object = json.dumps(interaction_json)
        with open(path + "Interactions.json", "w") as outfile:
            outfile.write(json_object)
  
    def perturb_edges(self, origin_filter = []):
        for edge in self.edges:
            edge.perturbed = not edge.fits_origins(origin_filter)
            for modification in edge.modifications:
                modification.perturbed = not modification.fits_origins(origin_filter)
        for node in self.nodes:
            node.sqrt_out_degree = math.sqrt(node.out_degree)
            node.sqrt_in_degree = math.sqrt(node.in_degree)
            
# Network Analysis

    def bfs(self, starting_nodes, directed = True):
        
        visited = set(starting_nodes)
        visited_edges = set()
        queue = deque(starting_nodes)

        # BFS traversal
        while queue:
            node = queue.popleft()
            for edge in node.incoming if directed else node.all_edges:
                visited_edges.add(edge)
                for neighbor in (set(edge.sources) | edge.modifiers) if directed else edge.all_nodes:# node.all_connected_nodes:            
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return (visited,visited_edges)
    
    def remove_disconnected_networks(self, node_filter = lambda node: node.submap, directed = True):
        
        start = time.process_time()

        visited,visited_edges = self.bfs([node for node in self.nodes if node_filter(node)], directed = directed)

        # print("perform bfs", time.process_time() - start)
        start = time.process_time()
        
        removed_edges = 0
        removed_nodes = 0

        for _hash,edge in [(k,v) for k,v in self.edges.items()]:
            if not edge.submap and edge not in visited_edges:
                for source in edge.sources:
                    source.outgoing.remove(edge)
                for target in edge.targets:
                    target.incoming.remove(edge)    
                for modification in edge.modifications:
                    for modifier in modification.modifiers:
                        modifier.modifications.remove(modification) 
                del self.edges[_hash]
                removed_edges += 1
            else:
                visited.update(edge.all_nodes)
                
        # print("remove edges", time.process_time() - start) 
        print("removed edges: ", removed_edges)
        start = time.process_time()

        for _hash,node in [(k,v) for k,v in self.nodes.items()]:
            if not node.submap and node not in visited:
                for parent in node.parents:
                    parent.subunits.remove(node)
                for subunit in node.subunits:
                    subunit.parents.remove(node)
                node.entity.nodes.discard(node)
                del self.nodes[_hash]
                removed_nodes += 1
        
        # print("remove nodes", time.process_time() - start)   
        print("removed nodes: ", removed_nodes)            
        
        for _hash,entity in [(k,v) for k,v in self.entities.items()]:
            if len(entity.nodes) == 0:
                del self.entities[_hash]  
                
        self.adjust_indices()
                
#     def get_interaction_of_nodes(self, node1, node2, integrate_enzymes = True, enzyme_feedback = True, origin_filter = []):
        
#         interaction_integers = []
        
#         for edge in node1.outgoing:
#             if not origin_filter or any([origin in origin_filter for origin in edge.origins]):
#                 catalyses = edge.is_catalyzed

#                 # add the source->target interaction only if there is no enzyme or enzyme is not integrated and node2 is a target of node1
#                 if (not integrate_enzymes or not catalyses) and node2 in edge.targets:
#                     interaction_integers.append(edge.edge_type_int)

#                 # add the source->enzyme interaction if enzymes should be integrated and node2 is an enzyme with node1 as a substrate
#                 if integrate_enzymes and any([node2 in catalysis.modifiers for catalysis in edge.is_catalyzed if not origin_filter or any([origin in origin_filter for origin in catalysis.origins])]):
#                     interaction_integers.append(1)            
            
#         # add the negative enzyme->source feedback for catalyses if enzyme_feedback is true and node1 is an enzyme with node2 as substrate
#         if enzyme_feedback:
#             interaction_integers += [-1 for edge in node2.outgoing if any([node1 in catalysis.modifiers for catalysis in edge.is_catalyzed if not origin_filter or any([origin in origin_filter for origin in catalysis.origins])])]
            
#         # add the modifier->target interaction if node1 is modifier of node2
#         for edge in node2.incoming:
#             if not origin_filter or any([origin in origin_filter for origin in edge.origins]):
#                 for modification in edge.modifications:
#                     if node1 in modification.modifiers:
#                         if modification.is_catalysis:
#                             return 1
#                         else:
#                             interaction_integers.append(modification.modification_on_target_int)
        
#         return np.sign(sum(interaction_integers, 0))
                
#     def as_matrix(self, integrate_enzymes = True, enzyme_feedback = True, origin_filter = [], reverse = False):
#         indices = list(enumerate(list(self.nodes)))
#         matrix = np.zeros(shape=(len(indices), len(indices)))
        
#         for i, source in indices:
#             for j, target in indices:
#                 matrix[j if reverse else i][i if reverse else j] = self.get_interaction_of_nodes(source, target, integrate_enzymes = integrate_enzymes, enzyme_feedback = enzyme_feedback, origin_filter = origin_filter)
#         return matrix
    
#     def as_adj_list(self, matrix = None, integrate_enzymes = True, enzyme_feedback = True, origin_filter = [], reverse = False):
#         if not matrix:
#             matrix = self.as_matrix(integrate_enzymes = integrate_enzymes, enzyme_feedback = enzyme_feedback, origin_filter = origin_filter, reverse = reverse)
#         nodes = list(self.nodes)
#         adj_list = {}
#         for i in range(matrix.shape[0]):
#             adj_list[nodes[i]] = [nodes[j] for j in range(matrix.shape[1]) if matrix[i][j]]
#         return adj_list

    def shortest_paths(self, origin_filter = []):        
        return scipy.sparse.csgraph.shortest_path(self.as_matrix(origin_filter = origin_filter), method='auto', directed=True, return_predecessors=False, unweighted=True, overwrite=False, indices=None)
    
    def as_networkx(self, origin_filter = []):
        return nx.from_numpy_array(self.as_matrix(origin_filter = origin_filter))

    def get_bipartite_graph(self):
        graph = defaultdict(dict)
        for edge in self.edges:
            has_modifiers = len(edge.modifications) > 0
            for source in edge.sources:
                for target in edge.targets:
                    if has_modifiers:
                        for modification in edge.modifications:
                            for modifier in modification.modifiers:
                                graph[source][modifier] = 1
                                graph[modifier][target] = modification.modification_on_target_int
                    else:
                        graph[source][target] = edge.edge_type_int
        return graph

    def find_connected_components_undirected(self):
        
        graph = self.get_bipartite_graph()
        visited = set()
        components = []

        def dfs(node, component, edges):
            if node not in visited:
                visited.add(node)
                component.append(node)
                for neighbor, interaction_type in graph.get(node, {}).items():
                    edge_representation = (node, interaction_type, neighbor)
                    edges.add(edge_representation)
                    dfs(neighbor, component, edges)
                # This additional loop handles the case where a node is a target but not a source
                for source, interactions in graph.items():
                    if node in interactions:
                        interaction_type = interactions[node]
                        edge_representation = (source, interaction_type, node)
                        edges.add(edge_representation)
                        dfs(source, component, edges)

        for node in graph:
            if node not in visited:
                component = []
                edges = set()
                dfs(node, component, edges)
                components.append((component, list(edges)))

        return components


    
    def get_influence_scores(self, phenotypes = [], submap_specific = True):
    
        if not phenotypes:
            phenotypes = self.phenotypes

        same_origin_phenotypes = defaultdict(list)
        start_time = time.time()
        for phenotype, origins in [(phenotype, tuple(phenotype.origins)) for phenotype in phenotypes]:
            same_origin_phenotypes[origins].append(phenotype)
        end_time = time.time()
        # print("Time taken for same_origin_phenotypes:", end_time - start_time)

        start_time = time.time()
        adjacency_list = self.as_adjacency_list(reverse=True, submap_specific = submap_specific)
        end_time = time.time()
        # print("Time taken for as_adjacency_list:", end_time - start_time)

        phenotype_influence_scores = {
            "paths": {},
            "values": {},
            "SPs": {}
        }

        start_time = time.time()
        for origins, related_phenotypes in same_origin_phenotypes.items():
            self.propangate_signal(n_steps=20, alpha=0.5, reverse=True,
                                   conditions=[{phenotype: 1} for phenotype in related_phenotypes],
                                   origin_filter=origins)

            origins = set(origins)
            filtered_adjacency_list = {source: {target: properties["skipped"] for target, properties in adjacency.items() if properties["origins"] & origins} for source, adjacency in adjacency_list.items()}

            for i, phenotype in enumerate(related_phenotypes):
                phenotype_influence_scores["values"][phenotype] = phenotype.get_influence_scores(filtered_adjacency_list, node_weights={node: np.sign(np.sum(node.signals[i])) for node in self.nodes})
                phenotype_influence_scores["paths"][phenotype] = phenotype.shortest_paths(filtered_adjacency_list)
        end_time = time.time()
        # print("Time taken for calculating values and paths:", end_time - start_time)

        filtered_adjacency_list = {source: {target: properties["type"] for target, properties in adjacency.items()} for source, adjacency in adjacency_list.items()}

        start_time = time.time()
        for phenotype in phenotypes:
            phenotype_influence_scores["SPs"][phenotype] = phenotype.shortest_path_length(filtered_adjacency_list)
        end_time = time.time()
        # print("Time taken for calculating SPs:", end_time - start_time)

        return phenotype_influence_scores
    
    def as_adjacency_list(self, reverse = False, submap_specific = False):
        
        adjacency_list = {} #{node:{target:{"skipped":[[]], "origins":set()} for target in self.nodes} for node in self.nodes}
        
        for edge in [edge for edge in self.edges if edge.submap] if submap_specific else self.edges:
            for node1,node2,skipped_node,_type,origins in edge.as_edge_pairs():
                source = node2 if reverse else node1
                target = node1 if reverse else node2
                if source not in adjacency_list:
                    adjacency_list[source] = {}
                if target not in adjacency_list[source]:
                    adjacency_list[source][target] = {"skipped":set(), "origins":set(), "type": _type}
                adjacency = adjacency_list[source][target]
                if skipped_node:
                    adjacency["skipped"].add(skipped_node)
                adjacency["origins"].update(origins)
                
        return adjacency_list 

#Boolean
    
    def set_initial_state(self, grid = None):
        if grid:
            self.grid = grid
            self.grid_size = self.grid[0] * self.grid[1]
            
        # templates return by nodes with static activity or for perturbation purposes
        self.false_template = np.full(self.grid_size, False)
        self.true_template = np.full(self.grid_size, True)
        self.zero_template = np.zeros(self.grid_size)
        self.ones_template = np.ones(self.grid_size)

        # storing the activity of nodes at each step
        # both are necessary, because nodes are evaluated iteratively. If the activity of a node is updated, another node still needs to refer to its previous acitivty
        self.previous_activities = np.zeros((len(self.nodes), self.grid_size))        
        self.current_activities = np.zeros((len(self.nodes), self.grid_size))
        
        # list of sparse arrays to store the change in activity for each node at each step compared to the previous step
        self.store_activities = []
        
        # Preallocated arrays that store data for the sparse array (reset at each step)
        self.sparse_row_indices = np.zeros(len(self.nodes) * self.grid_size)
        self.sparse_col_indices = np.zeros(len(self.nodes) * self.grid_size)
        self.sparse_data = np.zeros(len(self.nodes) * self.grid_size)
        
        # list of sparse arrays to store the change in perturabtion for each node at each step compared to the previous step
        self.store_perturbations = []
        
        self.previous_perturbation = {node.index:0 for node in self.nodes}    
                
        # Preallocated arrays that store data for the sparse array
        self.perturb_sparse_row_indices = []
        self.perturb_sparse_col_indices = []
        self.perturb_sparse_data = []

        # reset all nodes
        for node in self.nodes:
            node.perturbation = 0
            node.active.cache_clear()
            
        # initialize the "zeroth" step in which all nodes are updated to get their initial state
        self.activity_step(first_step = True)
        
    def activity_step(self, first_step = False):

        # To keep track of where we are in the preallocated arrays self.sparse_row_indices,self.sparse_col_indices,self.sparse_data
        current_idx = 0  

        # in the first step of a simulation all nodes need to be evaluated
        if first_step:
            nodes_to_eval = self.nodes
        # in following steps only updates nodes which with the possibility to change their state
        else:            
            # e.g. nodes whose activities are based on probabilities need to be updated every step
            nodes_to_eval = set(node for node in self.nodes if node.always_update)
            # otherwise update only targets of nodes whose activity has changed
            for node in self.nodes_with_changes | {node for node in self.nodes if node.delay or node.refill != None}:
                nodes_to_eval.update(node.boolean_targets)

        # node.active() results are cached by the @functools module
        # for nodes whose activity has changed in the previous step, it needs to be cleared
        nodes_to_clear_cache = self.nodes_with_changes | {node for node in self.nodes if node.always_update}
        for node in nodes_to_clear_cache:
            node.active.cache_clear()
            
        # keep track of nodes whose activity or perturbation changed
        self.nodes_with_changes = set()
        
        for node in nodes_to_eval:
            # previous_activity will be index in update_activity() anway so it gets returned to avoid another indexing
            previous_activity, new_activity = node.update_activity()
            
            # get all activity values that are different to the previous step
            diff = new_activity - previous_activity
            non_zero_positions = np.where(diff != 0)[0]
            
            # if there are changes add them to the np arrays that stored the data for the sparse arrays
            if len(non_zero_positions) > 0:
                self.nodes_with_changes.add(node)
                
                self.current_activities[node.index] = new_activity
                non_zero_values = diff[non_zero_positions]

                num_updates = len(non_zero_positions)

                self.sparse_row_indices[current_idx:current_idx + num_updates] = node.index
                self.sparse_col_indices[current_idx:current_idx + num_updates] = non_zero_positions
                self.sparse_data[current_idx:current_idx + num_updates] = non_zero_values

                current_idx += num_updates

        # Slice arrays up to current_idx
        row_indices = self.sparse_row_indices[:current_idx]
        col_indices = self.sparse_col_indices[:current_idx]
        data = self.sparse_data[:current_idx]  

        # Create coo_matrix from the activity array and indices        
        sparse_activity_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(len(self.nodes), self.grid_size), dtype=np.int32)
        self.store_activities.append(sparse_activity_matrix)
        
        # go through each node and check whether the perturbation has changed compared to previous step
        for node in self.nodes:
            prev_perturbation = self.previous_perturbation[node.index]
            
            # simple comparisons to check whether there are differences, i.e. if both are 0 there is none, etc.
            if isinstance(node.perturbation, np.ndarray):
                if isinstance(prev_perturbation, np.ndarray):
                    diff = node.perturbation - prev_perturbation
                else:
                    diff = node.perturbation
            else:
                if isinstance(prev_perturbation, np.ndarray):
                    diff = -prev_perturbation
                else:
                    continue

            # get all values that are different to the previous step
            non_zero_positions = np.where(diff != 0)[0]
        
            # if there are changes add them to lists that stored the data for the sparse arrays
            if len(non_zero_positions) > 0:
                non_zero_values = diff[non_zero_positions]
                # Update the lists
                self.perturb_sparse_row_indices.extend([node.index] * len(non_zero_positions))
                self.perturb_sparse_col_indices.extend(non_zero_positions)
                self.perturb_sparse_data.extend(non_zero_values)

            self.previous_perturbation[node.index] = node.perturbation
            node.perturbation = 0
 
        # Create coo_matrix from the different perturbation data
        sparse_perturbation_matrix = coo_matrix((self.perturb_sparse_data, (self.perturb_sparse_row_indices, self.perturb_sparse_col_indices)), 
                                                shape=(len(self.nodes), self.grid_size), 
                                                dtype=np.int32
                                               )
        self.store_perturbations.append(sparse_perturbation_matrix)

        self.perturb_sparse_row_indices.clear()
        self.perturb_sparse_col_indices.clear()
        self.perturb_sparse_data.clear()
        
        # assign current step as the previous one for the next step 
        self.previous_activities[:, :] = self.current_activities

    # restore the list of coo sparse matrices for activities and perturbations of a specific node into a 2D (step * position) np array 
    def restore_matrix_at_node(self,node, control = False):

        def restore_matrix(sparse_matrices):
            # Initialize with a zero matrix
            restored_matrix = np.zeros((len(sparse_matrices)+1,self.grid_size))
            # Iteratively sum COO matrices
            for i,coo in enumerate(sparse_matrices):
                restored_matrix[i+1] += restored_matrix[i] 
                # Extract the columns and values of the specific node (i.e. row in the array)
                node_rows = coo.row == node.index
                # Sum the values for the specific node
                for j, v in zip(coo.col[node_rows], coo.data[node_rows]):
                    restored_matrix[i+1][j] += v
                    
            return restored_matrix

        if control:
            return (restore_matrix(self.ctrl_store_activities),restore_matrix(self.ctrl_store_perturbations))
        else:
            return (restore_matrix(self.store_activities),restore_matrix(self.store_perturbations))
    
    # restore the list of coo sparse matrices for activities and perturbations of a specific position into a 2D (step * nodes) np array     
    def restore_matrix_at_pos(self,pos):

        col_idx= np.ravel_multi_index(pos, self.grid)        

        def restore_matrix(sparse_matrices):
            # Initialize with a zero matrix
            restored_matrix = np.zeros((len(sparse_matrices)+1, len(self.nodes)))
            # Iteratively sum COO matrices
            for i,coo in enumerate(sparse_matrices):
                restored_matrix[i+1] += restored_matrix[i] 
                # Extract the columns and values of the specific position (i.e. column in the array)
                node_cols = coo.col == col_idx
                # Sum the values for the specific position
                for j, v in zip(coo.row[node_cols], coo.data[node_cols]):
                    restored_matrix[i+1][j] += v
                    
            return restored_matrix

        return (restore_matrix(self.store_activities),restore_matrix(self.store_perturbations))
         
    def run_boolean_simulation(self, steps = 10, conditions = {}, perturbations = {}, pos = (0,0)): 
        
        self.set_initial_state()       

        perturbations = {node: param if isinstance(param, list) else evenDist(100, param) for node,param in perturbations.items() if node}   
        conditions = {node: param if isinstance(param, list) else evenDist(100, param) for node,param in conditions.items() if node}  

        for i in range (steps):
            
            for node,pert_list in conditions.items():
                node.perturb(pert_list[i%len(pert_list)])
                
            for node,pert_list in perturbations.items():
                perturbation = pert_list[i%len(pert_list)]
                if perturbation:
                    node.perturb(perturbation)
            
            self.activity_step()

    def correlation_analysis(self, conditions, perturbed_nodes, perturbation = -1, steps = 20, simulation_steps = 100, pos = (0,0)):
        
        results = {"nodes": perturbed_nodes, "perturbation": perturbation, "x values": [i for i in range(0,101,int(100/steps))], "correlations": {node:{"values":[]} for node in self.nodes}}

        print_progress_bar(0,1)
        for x_value in results["x values"]:
            perturbations = {}
            for perturbed_node in perturbed_nodes:
                perturbations[perturbed_node] = perturbation * x_value
            for node, activity in self.run_boolean_simulation(steps = simulation_steps, conditions = conditions, perturbations = perturbations, pos = pos).items():
                results["correlations"][node]["values"].append(activity)
        
            print_progress_bar(x_value,100)
            
        for node, correlation in results["correlations"].items():
            if len(results["x values"]) < 2:
                corr = (0,1)
            else:
                try:
                    corr = scipy.stats.pearsonr(np.array(results["x values"]), np.array(correlation["values"]))
                except:
                    corr = (0,1)
            correlation["pvalue"] = corr[1]
            correlation["correlation"] = corr[0]
        
        return results
    
    def display_correlation_results(self, correlation_results, x_label = None, y_nodes = [], plot_width = 10, n_columns = 2, file_name = ""):
        
        if len(y_nodes) == 1:
            n_columns = 1
        fig, axs = plt.subplots(1 + len(y_nodes) // n_columns, n_columns, figsize = (plot_width,4*(len(y_nodes) // n_columns + 1)))

        for i,node in enumerate(y_nodes):

            x=np.array(correlation_results["x values"])
            
            if n_columns == 1:
                ax = axs[i]
            else:
                ax = axs[i//n_columns,i%n_columns]
            
            correlation = correlation_results["correlations"][node]
            y=np.array(correlation["values"])

            coef = np.polyfit(x,y,1)
            poly1d_fn = np.poly1d(coef) 
            ax.plot(x, y, 'ko', x, poly1d_fn(x), '--k')

            pvaluetext = ""
            if correlation["pvalue"] < 0.001:
                pvaluetext = "<0.001"
            else:
                pvaluetext = str(round(correlation["pvalue"],3))
            
            y = (coef[1])
            x = 0
            ax.annotate("Δy = " + str(round(coef[0],3)) + "\nR = " + str(round(correlation["correlation"],2)) + "\np-value: " + pvaluetext, (x, y), color= 'k')


            ax.set_ylabel("Activity of " + node.fullname + " [%]",color='black' ,fontsize=10)

            if x_label == None:
                ax.set_xlabel(" & ".join([node.fullname for node in correlation_results["nodes"]]) + " " + ("Deficiency" if correlation_results["perturbation"] == -1 else "Activity") + " [%]", fontsize= 10)
            else:
                ax.set_xlabel(x_label, fontsize= 10)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)

        if file_name:
            plt.savefig(file_name + ".png", dpi=600, bbox_inches='tight', transparent=True)
    
    def __load_submap_image(self, files):

        for file in files:
            if file in self.files:
                with open(self.files[file]["path"], "rb") as f:                     
                    r = requests.post('https://minerva-service.lcsb.uni.lu/minerva/api/convert/image/CellDesigner_SBML:png', data=f.read() , headers={'Content-Type':'application/octet-stream'})
                    if r.status_code == 200:
                        img = Image.open(io.BytesIO(r.content)).convert("RGBA")#.resize(self.files[file]["size"])
                        self.submap_images[file] = img
                        return img
        return None
    
    def get_submap_image(self, file, max_width = 1000, scale = 1.0, zoom_on_node = None, zoom = 1.0, zoom_ratio = 0):
        
        if file in self.submap_images:                    
            img = self.submap_images[file]
        else:
            img = self.__load_submap_image([file])

        width_scale = max_width / ((img.width/(zoom*2)*2)*scale)
        scale *= width_scale if width_scale < 1 else 1
                       
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.Resampling.LANCZOS) 
             
        if zoom_on_node:
            position = list(zoom_on_node.positions[file])[0]
            img = self.__zoom_at(img, position[0]*scale, position[1]*scale, zoom, zoom_ratio)

        return (img, scale)
    
    def __zoom_at(self, img, x, y, zoom, ratio = 0.0):
        w, h = img.size
        if ratio:
            h = int(ratio * w)
        zoom2 = zoom * 2
        img = img.crop((x - w / zoom2, y - h / zoom2, 
                        x + w / zoom2, y + h / zoom2))
        return img # img.resize((w, h), Image.Resampling.LANCZOS)   
    
    def __get_submap_overlay(self, file, overlay_img, highlights = {}, cmap = cm.bwr, max_width = 1500, scale = 1.0, zoom_on_node = None, zoom = 1.0, zoom_ratio = 0, alpha = 0.6):

        overlay = overlay_img.copy() 
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        norm = colors.Normalize(vmin=-1, vmax=1, clip = True)
        color_map = cm.ScalarMappable(norm=norm, cmap=cmap)

        for node, node_color in highlights.items():
            if file in node.positions and node_color:
                if isinstance(node_color, numbers.Number):
                    node_color = tuple([int(255*x) for x in color_map.to_rgba(float(node_color), alpha = alpha)])
                for position in node.positions[file]:
                    x,y,w,h = [point*scale for point in position]
                    draw.rectangle([x, y, x + w, y + h], fill=node_color)

        if zoom_on_node:
            position = list(zoom_on_node.positions[file])[0]
            overlay = self.__zoom_at(overlay, position[0]*scale, position[1]*scale, zoom, zoom_ratio)

        return overlay
    
    
    def highlight_on_map(self, file, overlay_img = None, highlights = {}, img = None, **kwargs):

        if not img:
            img, scale = self.get_submap_image(file, **kwargs)
            kwargs["scale"] = scale

        if not img:
            return None
        
        if not overlay_img:
            overlay_img = Image.new('RGBA', tuple([int(side*kwargs["scale"]) for side in self.submap_images[file].size]), (255,255,255,0))
        
        overlay = self.__get_submap_overlay(file, overlay_img, highlights, **kwargs)

        return Image.alpha_composite(img, overlay).convert("RGB")
        
        
    def show_boolean_simulation(self, file, min_steps = 0, max_steps = 100, slider_step = 1, pos = (0,0), conditions = {}, alpha = 0.6, prevent_run = False, **kwargs):
        
        img,scale = self.get_submap_image(file, **kwargs)        
        kwargs["scale"] = scale
        
        
        overlay_img = Image.new('RGBA', tuple([int(side*kwargs["scale"]) for side in self.submap_images[file].size]), (255,255,255,0))
        
        if not prevent_run:
            _ = self.run_boolean_simulation(steps = max_steps, conditions = conditions)
        
        node_activities, node_perturbations = self.restore_matrix_at_pos(pos)
        
        step_images = {step:
            self.highlight_on_map(file, overlay_img = overlay_img, highlights = {node:node.activity_color(node_activities[step], node_perturbations[step], alpha = alpha) for node in self.nodes if file in node.positions}, img = img, **kwargs)
         for step in range(min_steps, max_steps + 1, slider_step)}
        
        def f(step):            
            
            return step_images[step]
             
        wg.interact(f, step=wg.IntSlider(min=min_steps,max=max_steps-1,step=slider_step));

        
# Signal Propangation   

    def relationship_matrix(self, origin_filter = [], reverse = False):  
        
        self.perturb_edges(origin_filter = origin_filter)
        
        # Lists to store the non-zero elements and their positions
        row_indices = []
        col_indices = []
        values = []

        for edge in [edge for edge in self.edges if not edge.perturbed]:
            for source in edge.sources:
                for target in edge.targets:
                    value = edge.edge_type_int / (source.sqrt_out_degree * target.sqrt_in_degree)

                    row_indices.append(source.index)
                    col_indices.append(target.index)
                    values.append(value)

            for modification in [modification for modification in edge.modifications if not modification.perturbed]:
                for modifier in modification.modifiers:
                    for target in modification.edge.targets:
                        value = modification.modification_on_target_int / (modifier.sqrt_out_degree * target.sqrt_in_degree)

                        row_indices.append(modifier.index)
                        col_indices.append(target.index)
                        values.append(value)

                    if modification.is_catalysis:
                        for source in modification.edge.sources:
                            value = -1 / (modifier.sqrt_out_degree * source.sqrt_out_degree)

                            row_indices.append(modifier.index)
                            col_indices.append(source.index)
                            values.append(value)

        # Create the COO matrix
        coo_mat = coo_matrix((values, (row_indices, col_indices)), shape=(len(self.nodes), len(self.nodes)))

        # Transpose if reversed is True
        if reverse:
            coo_mat = coo_mat.transpose()
    
        self.perturb_edges() 
        
        # Convert the COO matrix to CSR format. This will automatically sum the values with the same indices
        return coo_mat.tocsr().transpose()            

    def propangate_signal(self, n_steps = 20, alpha = 0.5, reverse = False, signal_duration = None, conditions = [], node_weights = {}, origin_filter = [], np_fnc = np.sum, progress = False):
            
        if not isinstance(conditions, list):
            conditions = [conditions]

        for i,condition in enumerate(conditions):
            if all(isinstance(key, Node) for key in condition.keys()):
                conditions[i] = {"starting_signals": condition}
        
        rules_sparse = self.relationship_matrix(origin_filter = origin_filter, reverse = reverse)

        node_weighting = np.zeros(len(self.nodes))
        for node,weight in node_weights.items():            
            node_weighting[node.index] = weight        
        weight_down = 2**(-abs(node_weighting))
        weight_up = 2**(abs(node_weighting))
        
        activities = np.zeros((len(conditions), n_steps+1, len(self.nodes)))        
        knockouts = np.ones((len(conditions), len(self.nodes)))
        permutation = np.full(len(conditions), False)
        cutoffs = np.zeros(len(conditions))
        signal_duration = np.full(len(conditions), n_steps if signal_duration == None else signal_duration)
        
        for k,condition in enumerate(conditions):
            for node,score in condition["starting_signals"].items():
                activities[k,0,node.index] = score
            if "ko_nodes" in condition:
                for node in condition["ko_nodes"]:            
                    knockouts[k,node.index] = 0 
            if "cutoff" in condition:
                cutoffs[k] = condition["cutoff"] 
            if "signal_duration" in condition:
                signal_duration[k] = condition["signal_duration"] 
            permutation[k] = condition.get("permutation")
      
        has_knockouts = not np.all(knockouts == 1)
        has_cutoff = not np.all(cutoffs == 0)
        has_weighting = not np.all(node_weighting == 0)
        
        cutoffs = cutoffs.reshape((-1, 1))

        processed_starting_activity = (1-alpha) * activities[:,0]

        start = time.time()
        for t in range(1,n_steps):
            previous_activity = activities[:,t-1] 
            
            if has_cutoff:
                previous_activity = np.where(np.abs(previous_activity) < cutoffs, 0, previous_activity)
            if has_weighting:
                previous_activity = np.where((np.sign(previous_activity) == np.sign(node_weighting)), previous_activity * weight_up, previous_activity * weight_down)
            if has_knockouts:
                previous_activity *= knockouts        

            activities[:,t] = alpha * rules_sparse.dot(previous_activity.T).T  
            
            samples_with_signal = signal_duration > t   
            activities[samples_with_signal,t] += processed_starting_activity[samples_with_signal]
                
            if progress:
                print_progress_bar(t,n_steps)
                
        end = time.time()
        # print("Signals",end - start)
        
        start = time.time()
        for node in self.nodes:
            node.signals = activities[:,:,node.index]
            # node.signals = np_fnc(activities[:,:,index_mapping[i]], axis = 1) if i in index_mapping else np.zeros(len(starting_activities))
            
        end = time.time()
        # print("AUC", end - start)
        return rules_sparse.toarray()
        
    def fdr_corrected_pvalues(self, index = 0, permutations = [], show_plot = False, node_filter = lambda node: True, hist_node = None):

        distances = np.array([])
        signals = np.array([])
        slopes = []
        intercepts = []
        shapiro_scores = []
        delta_signals = np.array([])
        
        filtered_nodes = set([node for node in self.nodes if node_filter(node) and node.signals is not None])
        nodes_with_signal = [node for node in self.nodes if node.signals is not None]
        
        def get_signal_distance(node_signals):
            if node_signals.ndim == 1:
                if np.all(node_signals == 0):
                    return None  
                return (
                    node_signals.sum(), 
                    (node_signals != 0).argmax()
                )
            else:
                mask = np.any(node_signals != 0, axis=1)
                node_signals = node_signals[mask]
                if node_signals.size == 0:
                    return None                
                return (
                    np.abs(node_signals.sum(axis=1)), 
                    (node_signals != 0).argmax(axis=1)
                )
        
        def get_delta_signals(raw_signals, raw_distances):
            # Take the logarithm of the summed_signal
            log_summed_signal = np.log(raw_signals)
            # Perform linear regression
            slope, intercept, _, _, _ = linregress(raw_distances, log_summed_signal)
            # Calculate the fitted values using the regression line
            fitted_values = slope * raw_distances + intercept
            # Calculate the differences between the original log signals and the fitted values
            return (log_summed_signal - fitted_values, slope, intercept)

        # Define the exponential decay function
        def exponential_decay(x, a, b):
            return a * np.exp(-b * x)
           
        node_pvalues = []
        node_distributions = {}
        for i,node in enumerate(filtered_nodes):
            sample_data = get_signal_distance(node.signals[index])
            permutation_data = get_signal_distance(node.signals[permutations])
            if sample_data and permutation_data:
                
                distr_signals, distr_distances = permutation_data
                ln_distr_signals = np.log(distr_signals)
                
                sample_signal, sample_distance = sample_data
                log_sample_signal = np.log(np.abs(sample_signal))
                
                # Normalize signals for each distance
                unique_distances = np.unique(distr_distances)
                normalized_signals = np.zeros_like(distr_signals)
                for d in unique_distances:
                    mask = distr_distances == d
                    normalized_signals[mask] = (ln_distr_signals[mask] - np.mean(ln_distr_signals[mask])) / np.std(ln_distr_signals[mask])

                # Normalize the input signal using the mean and standard deviation for its corresponding distance
                input_mask = distr_distances == sample_distance
                normalized_input_signal = (log_sample_signal - np.mean(ln_distr_signals[input_mask])) / np.std(ln_distr_signals[input_mask])

                # Calculate the Z-score for the normalized input signal against the combined normalized distribution
                z_score = (normalized_input_signal - np.mean(normalized_signals)) / np.std(normalized_signals)

                # Calculate the p-value for the given normalized signal
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
               
                if not np.isnan(p_value):
                    node_pvalues.append((node, p_value, sample_signal, sample_distance))
                    
                # Perform the Shapiro-Wilk test
                _, p_value = scipy.stats.shapiro(normalized_signals)
                shapiro_scores.append(p_value)
                
                signals = np.append(signals, distr_signals)
                distances = np.append(distances, distr_distances)
                print_progress_bar(i+1,len(filtered_nodes))
                
                node_distributions[node] = normalized_signals
                   
        return node_distributions
    
        pvalues_corrected = multipletests([x[1] for x in node_pvalues], method='fdr_bh')[1]

        start = time.process_time()
        # Sort distances and signals based on distances
        sorted_indices = np.argsort(distances)
        distances = distances[sorted_indices]
        signals = signals[sorted_indices]

        # Fit the exponential decay model to the data
        params, covariance = curve_fit(exponential_decay, distances, signals)
        a_fit, b_fit = params

        # Calculate the fitted exponential decay
        fitted_signals = exponential_decay(distances, a_fit, b_fit)

        # Transform the data by taking the natural logarithm
        ln_signals = np.log(signals) #np.sign(signals) * np.abs(np.log(np.abs(signals)))

        
        sample_data_x = []
        sample_data_y = []
        sample_data_s = []
        
        results = {}    
        
        for (node,pvalue,signal,distance),adj_pvalue in sorted(zip(node_pvalues, pvalues_corrected), key = lambda x: x[1]):        
            results[node] = (signal, pvalue, adj_pvalue, distance)
            sample_data_x.append(distance)
            sample_data_y.append(np.log(abs(signal)))
            sample_data_s.append(100 if adj_pvalue < 0.1 else 20)
            
        if show_plot:
            # Set a seaborn style
            sns.set_theme(style="whitegrid")

            # Plot the original data and fitted exponential decay
            fig = plt.figure(figsize=[18, 6])
            plt1 = fig.add_subplot(1, 3, 1)
            plt2 = fig.add_subplot(1, 3, 2)
            plt3 = fig.add_subplot(1, 3, 3)
            
            plt1.plot(distances, signals, 'o', label='Permutation Data', c = "lightblue")
            plt1.plot(distances, fitted_signals, '--', label='Fitted Exponential Decay', c = "darkred")
            plt1.set_title('Node Signal vs. Distance')
            plt1.set_xlabel('Distance')
            plt1.set_ylabel('Signal')
            plt1.legend()
                
            plt3.hist(shapiro_scores, bins = 100, density = True)

            plt3.set_ylabel('Frequency')
            plt3.set_xlabel('Shapiro–Wilk p-value')
            # plt3.set_xticks([])
            plt3.set_yticks([])
            
            # Plot the individual transformed data points
            plt2.plot(distances, ln_signals, 'o', alpha = 0.1, label='Transformed Permutation Data', c="lightblue")

            x_values = np.unique(distances)
            slope, intercept, _, _, _ = linregress(distances, ln_signals)

            # Initialize arrays to hold the min and max Y values
            y_min = np.full_like(x_values, float('inf'))
            y_max = np.full_like(x_values, float('-inf'))

            # Calculate Y values for each line and update the min and max arrays
            for slope, intercept in zip(slopes, intercepts):
                y_values = slope * x_values + intercept
                y_min = np.minimum(y_min, y_values)
                y_max = np.maximum(y_max, y_values)

            # Plot the shaded area
            plt2.fill_between(x_values, y_min, y_max, color='grey', label='Linear Regressions of Permutations', alpha=0.5)

            plt2.set_title('log-transformed Signal with 95% Confidence Interval')
            plt2.set_xlabel('Distance')
            plt2.set_ylabel('ln(Signal)')
            plt2.plot(sample_data_x, sample_data_y, 'o', label='Sample Data', c = "darkred", markersize = 3)# , s = sample_data_s)

            plt2.legend()
            leg = plt2.legend()
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

        # fig.savefig("plots/signal_transformation.png", dpi=600, bbox_inches="tight")
        return results

    #Data Mapping
    def map_dataframe(self, df, filter_null = True, node_filter = lambda node: node.type == "rna"):
        node_mapping = {node.name.lower():node for node in self.nodes if node_filter(node)}
        df = df.rename(index={index:node_mapping.get(str(index).lower()) for index in df.index},inplace=False).fillna(value = {col: 1 if col.endswith("_pvalue") else 0 for col in df.columns})#.dropna()
        return df[~df.index.isnull()] if filter_null else df
    
    
    #variant mapping:                    
    def create_genome_file(self, genome):
        self.transcripts = []
        node_index = defaultdict(list)
        for node in self.nodes: 
            node_index[node.name.lower()].append(node)
        mapped_data = {}
        for index,row in pd.read_csv(genome, sep="\t", encoding="ISO-8859-1", index_col = 0, header=None).astype(str).iterrows():
            nodes = node_index.get(index.lower())
            if nodes:
                for transcript in json.loads(row[1].replace("'", "\"").replace("False", "false").replace("True", "true")):
                    transcript["node"] = nodes
                    self.transcripts.append(transcript)

    def create_transcript_dictionary():
        self.transcript_index = {}
        for _id, t in enumerate(self.transcripts):
            if t["c"] not in self.transcript_index:
                self.transcript_index[t["c"]]  = IntervalTree()
                if t["e"] - t["s"] > 0:
                    self.transcript_index[t["c"]][t["s"]:t["e"]] = t
                
    def get_transcripts(chrom, pos): 
        # Query the tree for all intervals that overlap with the point
        intervals = t_dict[chrom][pos]
        # Extract original range and position from each interval
        return [interval.data[2] for interval in intervals]