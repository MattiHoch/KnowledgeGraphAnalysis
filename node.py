import time
import re
from collections import defaultdict, Counter, deque
import numpy as np
import math
from sklearn import metrics
import statistics
from scipy import stats
import hashlib
from functools import lru_cache
from scipy.sparse import csr_matrix, lil_matrix
import functools
import requests

def trim_node_name(name):
    return name.lower().replace("hsa-", "").replace("-5p", "").replace("-3p", "").replace("mir-", "mir")
        
class Entity:
    def __new__(cls, model, node):
            
            _hash = hashlib.sha256(trim_node_name(node.name).encode()).hexdigest()

            entity = model.entities.get(_hash)
            if entity:
                entity.origins.update(node.origins)
                entity.nodes.add(node)
                return entity
            entity = super(Entity, cls).__new__(cls)
            model.entities[_hash] = entity
            entity.hash = _hash
            entity.hash_string = str(_hash)
            return entity
    
    def __init__(self, model, node):       
        if hasattr(self, 'name'):
            return
        self.nodes = set([node])
        self.name = node.name
        self.model = model
        self.origins = set(node.origins)

class Node:
    def __new__(cls, model, name, nodetype = "Protein", states: tuple = (), subunits = (), family = False, compartment = "", initial = False, hypothetical = False, storage = 0, delay = 0, decay = 1, origins = [], map_ids = [], positions = {}, submap = False, references = []):
        if nodetype.lower() == "mirna":
            nodetype = "RNA"
 
        _hash = hashlib.sha256(str((
            trim_node_name(name) if len(subunits) == 0 else tuple(sorted(subunits)),
            nodetype.lower(), 
            compartment, 
            states, 
            hypothetical
        )).encode()).hexdigest()
    
        node = model.nodes.get(_hash)

        if node:
            node.origins.update(origins)
            for origin in origins:
                if origin in node.map_ids:
                    node.map_ids[origin].update(map_ids)
                else:
                    node.map_ids[origin] = set(map_ids)
            node.references.update(references)
            if storage > node.storage:
                node.storage = storage
            if delay > node.delay:
                node.delay = delay
            if decay != node.decay and decay != 1:
                node.decay = decay
            if submap:                
                node.submap = submap
            for file, pos in positions.items():
                if file not in node.positions:
                    node.positions[file] = set(pos)
                else:
                    node.positions[file].update(pos)
            return node
        
        node = super(Node, cls).__new__(cls)
        
        model.nodes[_hash] = node
        node.hash = _hash
        node.hash_string = str(_hash)
        return node
   
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['rule']  # Don't serialize the rule attribute
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.update_rule()  
        
    def __init__(self, model, name, nodetype = "Protein", states: tuple = (), subunits = (), family = False, compartment = "", initial = False, hypothetical = False, storage = 0, delay = 0, decay = 1, origins = [], map_ids = [], positions = {}, submap = False, references = []):           
        if hasattr(self, 'name'):
            return
        
        self.model = model
        self.name = name
        self.type = nodetype.lower()
        self.compartment = compartment.lower()
        self.incoming = set()
        self.outgoing = set()
        self.modifications = set()
        self.states = tuple(sorted(states))
        self.subunits = subunits
        self.parents = set()
        for subunit in subunits:
            subunit.parents.add(self)
        self.origins = set(origins)
        self.map_ids = {origin:set(map_ids) for origin in origins}
        self.family =  family  
        self.positions = positions
        self.submap = submap
        self.hypothetical = hypothetical
        self.references = set(references)
        
        self.signals = []
        self.starting_signal = 0

        self.initial_activity = 0
        self.storage = storage
        self.delay = int(delay)
        self.decay = int(decay)
        self.rule = None
        self.consumption = None
        self.refill = None
        self.refill_sources = set()
        self.boolean_targets = []
        
        self.perturbation = 0
        
        self.index = len(model.nodes) - 1
        
        self.active.cache_clear()
        self.entity = Entity(model, self)
        
    def __eq__(self, othr):
        return (isinstance(othr, type(self)) and hash(self) == hash(othr))

    def __hash__(self):
        return int(self.hash, 16)
    
    def __gt__(self, node2):
        if len(self.subunits) == 0 and len(node2.subunits) == 0:
            return (self.name > node2.name) if self.name != node2.name else (self.compartment.lower() > node2.compartment.lower()) if self.compartment.lower() != node2.compartment.lower() else  (",".join(self.states) > ",".join(node2.states))
            
        elif self.subunits == node2.subunits:
            return (self.compartment.lower() > node2.compartment.lower()) if self.compartment.lower() != node2.compartment.lower() else (",".join(self.states) > ",".join(node2.states))
        else:
            return self.subunits > node2.subunits
    
    @property
    def type_class(self):
        if self.type.lower() in ["protein", "receptor", "tf"]:
            return "PROTEIN"
        if "phenotype" in self.type.lower():
            return "PHENOTYPE"
        else:
            return self.type
    
    
    @property
    def string_index(self):
        return "m" + str(self.index)
    
    def numexpr_index(self, probabilistic = False):
        # return self.string_index + "_" + ("true" if probabilistic else "false")
        return "(" + "node" + str(self.index) + ".active(probabilistic=" + ("True" if probabilistic else "False") + ", source=source_node))"

    @property
    def fullname(self):
        return self.name + "(" + self.type + (", " + self.compartment if self.compartment else "") + ")" + "".join(["_" + state for state in self.states]) + ("(hypothetical)" if self.hypothetical else "") 
    @property
    def fullname_printable(self):
        return self.name + " (" + self.type.replace("_", " ").capitalize() + (", " + (self.compartment.capitalize() if self.compartment else "Blood")) + ")" + ((" (" + "".join([", " + state.capitalize() for state in self.states]) + ")") if self.states else "") + (" (currency)" if self.hypothetical else "")
    
    @property
    def minerva_name(self):
        return self.name + " (" + (self.compartment if self.compartment else "Blood") + ")"
       
    @property
    def in_degree(self):
        return len([edge for edge in self.incoming if not edge.perturbed]) + len(sum([edge.is_catalyzed for edge in self.outgoing if not edge.perturbed], []))
    @property
    def out_degree(self):
        return len([edge for edge in self.outgoing if not edge.perturbed]) + len([modification for modification in self.modifications if not modification.perturbed])
    
    @property
    def incoming_nodes(self):
        return set().union(*[edge.all_nodes for edge in self.incoming]) if self.incoming else set()
    
    @property
    def outgoing_nodes(self):
        return set().union(*[edge.all_nodes for edge in [self.outgoing] + [modification.edge for modification in self.modifications]]) if (self.outgoing or self.modifications) else set()
        
    @property
    def all_connected_nodes(self):
        return set().union({node for edge in self.all_edges for node in edge.all_nodes}, self.subunits, self.parents)
    
    @property
    def all_edges(self):
        return set().union(self.incoming, self.outgoing, [modification.edge for modification in self.modifications])
    
    @property
    def siblings(self):
        return list(self.entity.nodes)
    
    def extended_subunit_list(self):
        if self.family:
            return sum([subunit.extended_subunit_list() for subunit in self.subunits], [])
        else:
            return [self]
        
    def as_string(self, delimiter = " - "):
        return delimiter.join([
            self.name if len(self.subunits) == 0 else "[" + ",".join([subunit.name for subunit in self.subunits]) + "]",
            self.type,
            self.compartment,
            "[" + ",".join(self.states) + "]",
        ])

    def fits_origins(self, origin_filter):
        return True if not origin_filter or any([origin in origin_filter for origin in self.origins]) else False
    
    @property
    def simple_molecule(self):
        return self.type.lower() == "simple_molecule"
    
    #From Biohackathon2023
    def addReactomeSubunits(self, recursive = False):
        hsa_reactome_ids = [reference.replace("reactome:", "") for reference in self.references if reference.startswith("reactome:R-HSA")]
        if hsa_reactome_ids:
            res = requests.get('https://reactome.org/ContentService/data/complex/' + hsa_reactome_ids[0] + '/subunits?excludeStructures=false')
            if res.status_code == 200:
                subunits = res.json()
                for subunit in subunits:
                    name = subunit["name"][0]
                    node = Node(self.model, name, nodetype = self.type, compartment = self.compartment, origins = list(self.origins), submap = self.submap, references = [subunit["stId"]])
                    self.subunits += (node,)
                    if recursive:
                        node.addReactomeSubunits(recursive = recursive)

    
    
# Topology
         
    def shortest_path_length(self, adjacency_list):
        
        SPs = {}
        visited = set([self])
        queue = deque([(self, 1, 1)])

        while queue:
            node, sp_len, sp_type = queue.popleft()
            SPs[node] = sp_len * sp_type
            if node not in adjacency_list or not adjacency_list[node]:
                continue
            for neighbor,neighbor_type in adjacency_list[node].items():
                if neighbor not in visited and neighbor_type != 0:
                    visited.add(neighbor)
                    queue.append((neighbor, sp_len + 1, neighbor_type * sp_type))
        return SPs
    
    def shortest_paths(self, adjacency_list):
        
        paths = []
        visited = set([self])
        queue = deque([(self, [self])])

        while queue:
            node, path = queue.popleft()
            if len(path) > 1: 
                paths.append(reversed(path))
            if node not in adjacency_list or not adjacency_list[node]:
                continue
            for neighbor,skipped_nodes in adjacency_list[node].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    if skipped_nodes:
                        for skipped_node in skipped_nodes:
                            queue.append((neighbor, path + [skipped_node, neighbor]))
                            queue.append((skipped_node, path + [skipped_node]))
                    else:
                        queue.append((neighbor, path + [neighbor]))

        return paths


    def all_paths(self, reverse = False, submap_specific = True, shortest_paths = False):
               
        origin_filter = self.origins if submap_specific else []
        self.model.perturb_edges(origin_filter = origin_filter)
        
        visited = set()
        
        def recursive_paths(start, path=(), pathtype = 1):
            if start in path:
                return {}
            path = path  + (start,)
            paths = {path: pathtype}
            for edge in (start.incoming if reverse else start.outgoing):
                if not edge.perturbed:
                    for node in (edge.sources if reverse else edge.targets):
                        if node not in path:
                            if edge.modifications:
                                for modification in edge.modifications:
                                    if not modification.perturbed and modification.is_catalysis:
                                        for modifier in modification.modifiers:
                                            paths = {**paths, **recursive_paths(node, path = path + (modifier,), pathtype = pathtype * modification.modification_on_target_int)}
                                    else:
                                        for modifier in modification.modifiers:
                                            paths = {**paths, **recursive_paths(modifier, path = path, pathtype = pathtype * modification.modification_on_target_int)}
                            else:
                                paths = {**paths, **recursive_paths(node, path = path, pathtype = pathtype * edge.edge_type_int)}
            if reverse:
                for edge in [edge for edge in start.outgoing if not origin_filter or any([origin in origin_filter for origin in edge.origins])]:
                    for modification in edge.modifications: 
                        if not modification.perturbed and modification.is_catalysis:
                            for modifier in modification.modifiers:
                                paths = {**paths, **recursive_paths(modifier, path = path, pathtype = pathtype * -1)}
            else:
                for modification in self.modifications:
                    if not modification.perturbed and modification.modification.is_catalysis:
                        for source in modification.edge.sources:
                            paths = {**paths, **recursive_paths(source, path = path, pathtype = pathtype * modification.modification_int)}
            return paths
        
        if reverse:
            return {tuple(reversed(path)):_type for path,_type in recursive_paths(self).items()}
        else:
            return recursive_paths(self)
        
        self.model.perturb_edges()

# 2DEA

    def get_influence_scores(self, adjacency_list, node_weights = {}):
        
        # nodes = set(sum([[node] + list(adjacency.keys()) + sum([list(skipped) for skipped in adjacency.values()], []) for node,adjacency in adjacency_list.items() if adjacency], []))
        
        influence_scores = {}
        # print(adjacency_list)
        # print({node.name:{target.name:[skipped for skipped in skipped_nodes] for target,skipped_nodes in adjacency.items()} for node,adjacency in adjacency_list.items()})

        all_paths = set()

        nodes_on_paths = defaultdict(set)
        paths_with_node = defaultdict(int)
        visited = defaultdict(bool)
        
        # count = defaultdict(int)

        def dfs(node, path):
                        
            nodes_on_paths[node].update(path)

            if visited[node]:
                return paths_with_node[node]
            if node not in adjacency_list or not adjacency_list[node]:  # If no neighbors, it's a leaf node, so there's only one path (itself)
                paths_with_node[node] += 1
                return paths_with_node[node]
            
            if node in path:  # cycle detected
                return paths_with_node[node]

            visited[node] = True

            for neighbor,skipped_nodes in adjacency_list[node].items():
                if skipped_nodes:
                    for skipped_node in skipped_nodes:
                        nodes_on_paths[skipped_node].update(path)
                        skipped_paths = dfs(neighbor, path + [skipped_node, node])
                        paths_with_node[node] += skipped_paths
                        paths_with_node[skipped_node] += skipped_paths                        
                else:
                    paths_with_node[node] += dfs(neighbor, path + [node])

            return paths_with_node[node]

        # total paths is all paths from start
        total_paths = dfs(self, [])
        
        if not total_paths:
            return {}
        
        nodes_on_paths = {node:len(nodes) for node,nodes in nodes_on_paths.items() if nodes and node != self}
        all_nodes = len(nodes_on_paths)

        
        if not node_weights:
            node_weights = self.signal_effects(reverse = True)
        
        for node,nodes_on_paths_number in nodes_on_paths.items():
            influence_scores[node] = paths_with_node[node] / total_paths + nodes_on_paths_number / all_nodes
   
        if influence_scores:
            max_score = max(influence_scores.values())
            if max_score:
                return {node:score*node_weights[node]/max_score for node,score in influence_scores.items()}
            else:
                return {}
        else:
            return influence_scores


# Boolean

    # creating a lambda function for self.rule 
    # self.refill and self.consumption are very special cases and not important for the base principle
    def update_rule(self):
        self.boolean_expr = ""
        self.always_update = False
        
        if self.subunits and self.family:
            self.rule = lambda: self.model.false_template
            self.consumption = lambda: self.model.false_template
        else:
            
            nodes = list(self.model.nodes.values())                
            def map_numexpr_nodes(expr_string):
                pattern = r"node(\d+)."
                matches = re.findall(pattern, expr_string)
                # Create a dictionary to store node-boolean associations
                mapping = {"node"+match: nodes[int(match)] for match in matches}  
                return mapping

            edge_lambdas = {1:[], -1:[]}
            refills = []            
            self.refill_sources = set()
            for edge in self.incoming:
                edge_type,edge_lambda = edge.as_numexpr_string()
                if edge_lambda and edge_type != 0:
                    if edge.refill:
                        self.refill_sources.update(edge.basesources)
                        refills.append(edge_lambda)
                    else:
                        edge_lambdas[edge_type].append(edge_lambda)
            if len(refills) > 0:
                expr_string = " | ".join(refills)
                node_mapping = map_numexpr_nodes(expr_string)
                node_mapping["source_node"] = self
                self.refill = eval("lambda: " + expr_string, node_mapping)
            else:
                self.refill = None
            # edges that reduce the sotrage of a source element
            if self.storage:
                consumptions = []
                for edge in [edge for edge in self.outgoing if edge.consumption]:
                    edge_type,edge_lambda = edge.as_numexpr_string()
                    if edge_lambda and edge_type != 0:
                        consumptions.append((edge_lambda, edge.consumption)) 
                if len(consumptions) > 0:
                    expr_string = "sum([" + " , ".join([str(consumption) + " * " + edge_lambda for edge_lambda, consumption in consumptions]) + "])"
                    node_mapping = map_numexpr_nodes(expr_string)
                    node_mapping["source_node"] = self
                    self.consumption = eval("lambda: " + expr_string, node_mapping)
                else:
                    self.consumption = None
                    
            # create a string that performs logical operations on the node.active() (returns a  1D (reshaped from a 2D grid) boolean np array) for incoming nodes
            # sample string: “~ (node206.active() | node187.active()) & ((node167.active() & node167.active()) |  node207.active())”
            # nodes in the string are then mapped to node objects by the node.index
            numexpr_string = ""
            if len(edge_lambdas[-1]) > 0:
                numexpr_string += "~(" + " | ".join([expr for expr in edge_lambdas[-1]]) + ")"
            if len(edge_lambdas[1]) > 0:
                if numexpr_string:
                    numexpr_string += " & "
                numexpr_string += "(" + " | ".join(edge_lambdas[1]) + ")"
            self.boolean_expr = numexpr_string
            if numexpr_string:
                node_mapping = map_numexpr_nodes(numexpr_string)
                node_mapping["source_node"] = self
                for node in node_mapping.values():
                    node.boolean_targets.append(self)
                # converting the string into a lambda function
                self.rule = eval("lambda: " + numexpr_string, node_mapping)
            else:
                self.rule = lambda: (self.model.true_template if self.refill == None else self.model.false_template)
                
            if self.delay or self.storage or self.refill or self.consumption or ("probabilistic=True" in numexpr_string):
                self.always_update = True
        
    
    def perturb(self, perturbation):        

        # track nodes with changes in their activities
        # a comparison of whether there has been an actual change in self.perturbation is currently omitted as it is probably more extensive
        self.active.cache_clear()
        self.model.nodes_with_changes.add(self)  
        
        # self.perturbation is 0 by default (meaning no peturbation in all positions) to save storage
        # only converted to a 1D (reshaped from a 2D grid) np array when there is a perturbation
        if not isinstance(self.perturbation, np.ndarray):
            if not isinstance(perturbation, np.ndarray):
                if perturbation == 0:
                    self.perturbation = 0
                    return
            self.perturbation = self.model.zero_template.copy()  

        self.perturbation[:] = perturbation
        
    def perturb_at(self, perturbation, pos = (0,0)):
        
        self.active.cache_clear()
        self.model.nodes_with_changes.add(self)  
        
        if not isinstance(self.perturbation, np.ndarray):
            self.perturbation = self.model.zero_template.copy()  

        self.perturbation[pos] = perturbation
        
    def get_activity(self, pos = (0,0)):
        pos_idx= np.ravel_multi_index(pos, self.model.grid)        
        activity, perturbation = self.model.restore_matrix_at_node(self)
        if self.storage:
            activity /= self.storage
        activity = np.where(perturbation == 1, 1, activity)
        if len(self.refill_sources) > 0:
            for refill_node in self.refill_sources:
                refill_activity, refill_perturbation = self.model.restore_matrix_at_node(refill_node, control = control)
                activity = np.where((refill_activity > 0 & refill_perturbation != -1) | (refill_perturbation == 1), 1, activity)   
        activity = np.where(perturbation == -1, 0, activity)
        return activity[:,pos_idx]
        
    # evaluating self.rule() and rerturn the old and new activities as 1D (reshaped from a 2D grid) integer np arrays
    def update_activity(self):
        previous_activity = self.model.previous_activities[self.index]
        if self.storage:
            # # times 2 so that 0 becomes -1 and 1 stays 1
            delta_activity = 1*self.rule()
            delta_activity -= (~delta_activity.astype(bool))*self.decay
            new_activity = previous_activity + delta_activity
            if self.consumption:
                new_activity -= self.consumption()#.reshape(self.model.grid_size)
            new_activity = np.maximum(new_activity, 0)
            new_activity = np.minimum(new_activity, self.storage)
        else:
            new_activity = self.rule().astype(int)

        return (previous_activity, new_activity)

    # this is relevant only for nodes that have delay in propagating their signal
    # this function assess a nodes activity for any previous step by restoring the activity matrix from the sparse coo matrices stored in the self.model.store_activities list
    def active_at_step(self, at_step, delta = False):

        if delta:
            at_step = len(self.model.store_activities) - at_step
            
        if at_step < 0:
            return self.model.false_template, 0

        # return all stored sparse data for the current node
        def get_coo_row(coo):
            selector = coo.row == self.index
            data = coo.data[selector]
            col = coo.col[selector]
            return data,col
        
        def restore_matrix(restoring_matrix):
            # Initialize with a zero matrix
            restored_matrix = np.zeros(self.model.grid_size)
            
            # Add the differences up to the desired step
            # as most coo array will be very sparse, manually iterating through data and adding values is much more computational efficient than converting into a dense array
            for coo in restoring_matrix[:at_step]:
                data, cols = get_coo_row(coo)
                # Sum the desired row from each CSR matrix
                for col,val in zip(cols,data):
                    restored_matrix[col] += val

            return restored_matrix

        return (restore_matrix(self.model.store_activities), restore_matrix(self.model.store_perturbations))


    # returns node activity as a 1D (reshaped from a 2D grid) Boolean np array from self.model.previous_activities masked by self.perturbation
    # the result is cached to prevent multiple executions in a single step and also prevent reevaluation in subsequent steps if activity didn't change
    # Cache will be reset when incoming nodes change their state (happens in the Model.activtiy_step() function)
    @functools.lru_cache(maxsize=None)
    def active(self, source = None, probabilistic = False):

        if self.delay:
            if self.delay > len(self.model.store_activities):
                return self.model.false_template                    
            activity, perturbation = self.active_at_step(self.delay, delta = True)
        else:
            activity = self.model.previous_activities[self.index]
            perturbation = self.perturbation

        if probabilistic:
            np.random.seed(self.model.seed + self.model.step + self.index)
            activity = np.random.rand(self.model.grid_size) <= (activity / (self.storage if self.storage else 1))
        else:
            activity = activity.astype(bool)
        
        if self.refill and (not source or source not in self.refill_sources):  
            activity = (activity | self.refill()).astype(bool)
        
                    
        if isinstance(perturbation, np.ndarray):
            activity = np.where(perturbation == 1, True, np.where(perturbation == -1, False, activity))

        
        return activity

    # simply convert a node's activity from a given acitvity and perturbation array to a hex color
    # required for visualizations
    # normalized between -1 and 1: -1 = Blue, 0 = White, 1 = Red 
    def activity_color(self, activities, perturbations, alpha = 0.5, rgba = True, return_zero = None):
        alpha =  int(255*alpha)
        activity = activities[self.index]
        perturbation = perturbations[self.index]

        if perturbation == -1:
            return (40, 40, 40, 180)
        elif perturbation == 1:
            return (255, 0, 0, alpha)
        activity = activity / (self.storage if self.storage else 1)
        if len(self.refill_sources) > 0 and not activity:
            if any((activities[node.index] or perturbations[node.index] == 1) and perturbations[node.index] != -1 for node in self.refill_sources):
                activity = 1
        activity_color = int(255 * (1-(activity)))
        activity_color = (255, activity_color, activity_color, alpha)

        return (activity_color if rgba else '#{:02x}{:02x}{:02x}{:02x}'.format(*activity_color)) if activity else (return_zero + (alpha,) if return_zero else return_zero) 

    def current_activity_color(self, alpha = 0.5, pos = (0,0), rgba = True, return_zero = None):
        alpha =  int(255*alpha)
        if isinstance(self.perturbation, np.ndarray):
            if self.perturbation[pos] == -1:
                return (40, 40, 40, 180)
            elif self.perturbation[pos] == 1:
                (255, 0, 0, alpha)
        activity = self.model.current_activities[pos] / (self.storage if self.storage else 1)
        if self.refill and not activity and self.refill(step_number)[pos]:
            activity = 1
        activity_color = int(255 * (1-(activity)))
        activity_color = (255, activity_color, activity_color, alpha)

        return (activity_color if rgba else '#{:02x}{:02x}{:02x}{:02x}'.format(*activity_color)) if activity else (return_zero + (alpha,) if return_zero else return_zero) 
    
    def activity_colors(self, step_number = -1, alpha = 0.5):
        perturbations = np.array(self.perturbations[step_number])
        activities = np.array(self.activities[step_number]) / (self.storage if self.storage else 1)

        # Initialize the color array as a 4D array (rgba)
        color = np.zeros(perturbations.shape + (4,), dtype=int)

        # mask for perturbations
        mask = perturbations != 0

        # set color for perturbations
        color[mask, :] = np.where(perturbations[mask, np.newaxis] == -1, [40, 40, 40, 180], [255, 0, 0, int(255*alpha)])

        # set color for activities
        mask = ~mask
        activity_color = 255 * (1 - activities[mask])

        # Note that np.newaxis is used to match the dimensions for broadcasting
        color[mask, :] = np.array([255, activity_color, activity_color, int(255*alpha)])[:, np.newaxis]

        # convert rgba to hexadecimal (for entire array)
        hex_colors = np.apply_along_axis(lambda rgb: '#%02x%02x%02x%02x' % (rgb[0],rgb[1],rgb[2],rgb[3]), axis=-1, arr=color)

        return hex_colors
        
    def print_boolean_rule(self):
        edge_boolean_strings = {1:[], -1:[]}
        for edge in self.incoming:
            edge_type,edge_lambda = edge.as_boolean_string()
            if edge_type != 0:
                edge_boolean_strings[edge_type].append(edge_lambda)
        
        boolean_string = ""
        if len(edge_boolean_strings[1]) > 0:
            if len(edge_boolean_strings[1]) == 1:
                boolean_string += edge_boolean_strings[1][0]
            else:
                boolean_string += "(" + " OR ".join("(" + edge_string + ")" for edge_string in edge_boolean_strings[1]) + ")"
        if len(edge_boolean_strings[-1]) > 0:
            if boolean_string:
                boolean_string += " AND "
            boolean_string += " AND ".join(["NOT (" + edge_string + ")" for edge_string in edge_boolean_strings[-1]])
        
        return boolean_string
        
# Signal Transduction

    def signal_at_step(self, step):

        return self.signals[step]
    
    def get_reverse_static_signals(self):

        reverse_signals = np.zeros(len(self.model.nodes))
             
        for edge in self.outgoing:       
            for modification in edge.modifications:
                if not modification.perturbed and modification.is_catalysis:
                    for modifier in modification.modifiers:
                        reverse_signals[modifier.index] += -1 / (self.sqrt_out_degree*modifier.sqrt_out_degree)
                         
        for edge in self.incoming:
            if not edge.perturbed:
                for source in edge.sources:
                    reverse_signals[source.index] += edge.edge_type_int / (self.sqrt_in_degree*source.sqrt_out_degree)
            for modification in edge.modifications:
                if not modification.perturbed:
                    for modifier in modification.modifiers:
                        reverse_signals[modifier.index] += modification.modification_on_target_int/ (self.sqrt_in_degree*modifier.sqrt_out_degree)
        
        return reverse_signals


    def get_forward_static_signals(self):

        start = time.process_time()
        
        forward_signals = np.zeros(len(self.model.nodes))
        
        for edge in [edge for edge in self.outgoing if not edge.perturbed]:
            for target in edge.targets:
                forward_signals[target.index] += edge.edge_type_int / (self.sqrt_out_degree*target.sqrt_in_degree)

                        
        for modification in [modification for modification in self.modifications if not modification.perturbed]:
            for target in modification.edge.targets:
                forward_signals[target.index] += modification.modification_on_target_int / (self.sqrt_out_degree*target.sqrt_in_degree)
            if modification.is_catalysis:
                for source in modification.edge.sources:
                    forward_signals[source.index] += -1/(self.sqrt_out_degree*source.sqrt_out_degree)

        return forward_signals
    
    
    def update_signal(self, alpha = 0.5, **kwargs):
                
        current_step = len(self.signals) - 1

        self.signals.append(self.signal_rule(current_step, **kwargs) * (1-alpha) + self.starting_signal * alpha)

    # @property
    # def signal_auc(self, signal_number = 0):
    #     signal = self.signals[signal_number]
    #     return metrics.auc(range(len(signal)),signal)
    
    def signal_effects(self, submap_specific = True, reverse = False):
        
        self.model.propangate_signal(n_steps = 20, alpha = 0.5, reverse = reverse, starting_activities = {self:1}, origin_filter = self.origins if submap_specific else [])
        
        final_scores = {}
        
        for node in self.model.nodes:
            final_scores[node] = node.signals[0]
        
        max_score = max(final_scores.values(), default = 1)
        
        return {node:score/max_score for node,score in final_scores.items()}
    
    def p_value(self, value_index = 0):
    
        observed_value = self.signals[value_index]
        permuted_values = np.delete(self.signals, value_index)
        
        # mirror the permuted values
        permuted_values = np.concatenate([permuted_values, -permuted_values])

        # fit a normal distribution to the permuted values
        mu, std = stats.norm.fit(permuted_values)

        # calculate the cumulative density function (CDF) value for the observed value
        cdf_value = stats.norm.cdf(observed_value, mu, std)

        # for a one-tailed test (greater):
        p_value = 2 * min(cdf_value, 1 - cdf_value)
        
        return p_value if not np.isnan(p_value) else 1