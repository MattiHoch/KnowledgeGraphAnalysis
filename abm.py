import math
from os import path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import math
import IPython
import random
import time
from scipy.ndimage import convolve
import re 

try:
    from app.network_data.basic_functions import *
    from app.network_data.read_sbml import *
except:
    from network_data.basic_functions import *
    from network_data.read_sbml import *
    
class ABM():
    
    def __init__(self, project, agentmaps = [], outsidemaps = [], grid_size = 1, agents_per_quadrant = 1, seed = 0, image = ""):
        
        self.project = project
        self.size = grid_size
        
        if not seed:
            seed = random.randint(0, 10000)
            
        self.seed = seed
        
        quadrant_size = math.ceil(grid_size / agents_per_quadrant)
        
        if image:
            image_size = (self.size, self.size)
            userImage = Image.open(image)
            userImage = userImage.resize(image_size) ### EDITED LINE
        
        lower = 0.4
        upper = 1
        sigma = 0.1
        mu = 0.9
        
        np.random.seed(self.seed)
        mus = scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=quadrant_size**2)

        mu_lim = 0.05
        sigma = 0.01

        quadrant_probabilties = []
        for i in range(quadrant_size):
            quadrant_probabilties.append([])
            for j in range(quadrant_size):
                mu = mus[i*quadrant_size + j]
                lower = mu - mu_lim
                upper = mu + mu_lim
                np.random.seed(self.seed)
                quadrant_probabilties[i].append(scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=agents_per_quadrant**2))
                
        self.agents = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                if not image or any([rgb != 255 for rgb in userImage.getpixel((x, y))]):
                    self.agents[x][y] = quadrant_probabilties[x//agents_per_quadrant][y//agents_per_quadrant][(x%agents_per_quadrant) * agents_per_quadrant + y%agents_per_quadrant]

                    #"neighbors": self.get_neighbours(x,y),
                    
#         for agent in self.agentpositions.values():
#             agent["neighbors"] = [neighbor for neighbor in agent["neighbors"] if neighbor in self.agentpositions]

        self.agent_model = create_model(self.project, files = agentmaps, grid = (grid_size,grid_size), compartment_specific = True, seed = self.seed)
        self.outside_model = create_model(self.project, files = outsidemaps, compartment_specific = True, seed = self.seed)

        self.agents = self.agents.reshape(self.agent_model.grid_size)
            
        self.blood_node_pairs = {node:self.outside_model.nodes[_hash] for _hash,node in self.agent_model.nodes.items() if _hash in self.outside_model.nodes.keys()}
        self.liver_blood_nodes = [node for node in self.agent_model.nodes if node.compartment == ""]
        
        self.tag = self.agent_model.get_node_from_name("TAG")
        self.vldl = self.agent_model.get_node_from_name("VLDL")
        self.fa = self.agent_model.get_node_from_name("free fatty acids", compartment="")             
        self.food = self.outside_model.get_node_from_name("food intake")
        
    def get_neighbours(self, x, y):
        potential_neighbours = [(x-1, y-1), (x-1, y), (x-1, y+1), 
                                (x, y-1), (x, y+1), 
                                (x+1, y-1), (x+1, y), (x+1, y+1)]
        neighbours = [(nx, ny) for nx, ny in potential_neighbours if 0 <= nx < self.size and 0 <= ny < self.size]
        return neighbours
    
    def run_abm(self, steps = 100, perturbednodes = [], food_prob = 3*[1] + 5*[-1], conditions = {}, socketio = None, progress = True, aggregations = []):
        
        self.steps = steps

        self.agent_model.set_initial_state()
        self.outside_model.set_initial_state()

        self.food_prob = food_prob         
        
        self.perturbed_nodes = {
            "agent": self.agent_model.get_nodes_from_names(perturbednodes),
            "outside": self.outside_model.get_nodes_from_names(perturbednodes)
        }        

        measured_times = []
        for step in range(steps): 
            measured_times.append(self.step(step, steps, conditions = conditions, aggregations = aggregations))
            if progress:
                print_progress_bar(step+1, steps) 
            if socketio:
                socketio.emit('progress', {'id': 'abm_progress', 'percent': (step+1) / steps})
            
        print("Agent Step Time:" + str(sum([x[0] for x in measured_times])))
        print("Boolean Time:" + str(sum([x[1] for x in measured_times])))

    def step(self, current_step, steps, conditions = {}, aggregations = []):

        for node in self.perturbed_nodes["outside"]:
            node.perturb(-1)
            
        self.food.perturb(self.food_prob[current_step%len(self.food_prob)]) 
        np.random.seed(self.seed + current_step)
        random_numb = np.random.random()
        for perturbed_node,perturbation_score in conditions.items():            
            for node in perturbed_node.extended_subunit_list():
                node.perturb(0 if random_numb < perturbation_score else -1)
            
        self.outside_model.activity_step()
        
        start = time.time()
        
        pos_perturbed_array = np.full(self.agent_model.grid_size, 1)
        neg_perturbed_array = np.full(self.agent_model.grid_size, -1)
        
        np.random.seed(self.seed + current_step)
        random_array = np.random.rand(self.agent_model.grid_size)  # Generate random floats between 0 and 1

            
        nutrient_array = random_array <= self.agents

        for liver_node in self.liver_blood_nodes:
            if liver_node in self.blood_node_pairs:
                outside_node = self.blood_node_pairs[liver_node]
                if outside_node.boolean_expr != "":
                    intestine_node_activity = outside_node.active()
                    perturbation = -1 if liver_node.boolean_expr == "" and liver_node.refill == None else 0
                    if liver_node.simple_molecule:
                        liver_node.perturb(np.where(intestine_node_activity, np.where(nutrient_array, 1, perturbation), perturbation))
                    else:
                        liver_node.perturb(np.where(intestine_node_activity, pos_perturbed_array, perturbation))
            elif liver_node.boolean_expr == "":
                liver_node.perturb(-1)         

        kernel = np.ones((3, 3), dtype=np.int32)  # 3x3 kernel
        
        for aggregate_from, aggregate_to in aggregations:
            # aggregated_array = convolve(np.where(~self.vldl.perturbed_activities[-1], self.tag.activities[-1], 0), kernel, mode='constant') / 9 / self.tag.storage
            aggregated_array = (convolve(self.agent_model.current_activities[aggregate_from.index].reshape(self.agent_model.grid), kernel, mode='constant') / 9 / aggregate_from.storage).reshape(self.agent_model.grid_size)
            np.random.seed(self.seed + len(self.agent_model.store_activities))
            perturbation_array = (np.random.rand(self.agent_model.grid_size) < aggregated_array).astype(int)
            aggregate_to.perturb(perturbation_array)
        for node in self.perturbed_nodes["agent"]:
            node.perturb(-1)  

            
            
        end1 = time.time()  - start
        start = time.time()
        self.agent_model.activity_step()
        end2 = time.time() - start
        
        random.seed(self.seed + current_step)                    
        for liver_node, outside_node in self.blood_node_pairs.items():
            if outside_node.boolean_expr == "":
                mean_activity = np.mean(liver_node.active())
                if random.uniform(0, 1) <= mean_activity:
                    # print(outside_node.name, mean_activity)
                    outside_node.perturb(1)
                else:
                    outside_node.perturb(-1)
                    
        
        return((end1, end2))   
    
    def show_agents(self, file, node, alpha = 0.5, **kwargs):
        
        def extract_numbers(s):
            return list(map(int, re.findall(r'\d+', s)))
        
        agent_model = create_model(self.project, files = [file], id_as_name = True)
        node_postion_dict = {node:np.ravel_multi_index(tuple(extract_numbers(node.name)), self.agent_model.grid) for node in agent_model.nodes}
 
        img,scale = agent_model.get_submap_image(file, **kwargs)        
        kwargs["scale"] = scale
        
        overlay_img = Image.new('RGBA', tuple([int(side*kwargs["scale"]) for side in agent_model.submap_images[file].size]), (255,255,255,0))
        
        node_activities, node_perturbations = self.agent_model.restore_matrix_at_node(node)
        if node.storage:
            node_activities /= node.storage
        
        if len(node.refill_sources) > 0:
            for refill_node in self.refill_sources:
                refill_node_activities, refill_node_perturbations = self.agent_model.restore_matrix_at_node(refill_node)
                node_activities = np.where(((refill_activity > 0) & (refill_perturbation != -1)) | (refill_perturbation == 1), 1, activity)                
        node_activities = np.where(node_perturbations == 1, 1, node_activities)
        
        print(node_activities.max())
                
        # Start with a white hex color array
        node_colors = np.full(node_activities.shape, '#ffffff{:02x}'.format(int(255*alpha)))
        # Create a mask for non-zero values
        non_zero_mask = node_activities != 0
        # Extract non-zero values
        non_zero_values = node_activities[non_zero_mask]

        none_zero_colors = array_to_hex(non_zero_values, a = alpha)        
                               
        # Assign these hex values back to the original array using the mask
        node_colors[non_zero_mask] = none_zero_colors
        
        pert_color = '#{:02x}{:02x}{:02x}{:02x}'.format(*(40, 40, 40, int(255*alpha)))        
        node_colors = np.where(node_perturbations == -1, pert_color, node_colors)    
        node_colors = np.where(node_perturbations == 1, "#880000", node_colors)

        step_images = {step:
            agent_model.highlight_on_map(file, overlay_img = overlay_img, highlights = {node:node_colors[step][node_postion_dict[node]] for node in agent_model.nodes}, img = img, **kwargs) for step in range(0, self.steps + 1, 1)}
        
        def f(step):                        
            return step_images[step]
                             
        wg.interact(f, step=wg.IntSlider(min=0,max=self.steps-1,step=1));
        
    def get_agents(self, node, rgba = False, comparative = False, **kwargs):
        
        def weight_activity(control):
            activity, perturbation = self.agent_model.restore_matrix_at_node(node, control = control)
            if node.storage:
                activity /= node.storage
            activity = np.where(perturbation == 1, 1, activity)
            if len(node.refill_sources) > 0:
                for refill_node in node.refill_sources:
                    refill_activity, refill_perturbation = self.agent_model.restore_matrix_at_node(refill_node, control = control)
                    activity = np.where(((refill_activity > 0) & (refill_perturbation != -1)) | (refill_perturbation == 1), 1, activity)
            activity = np.where(perturbation == -1, 0, activity)
            return activity
        
        activity = weight_activity(False)
        
        if comparative:
                activity -= weight_activity(True)
            
            # in in the JS plugins an integer percentage value is used because decimals in javascript are strange hence * 100
            # in the JS plugins color gradients are precalculated in increments of 0.05 to improve speed, thus the activities are rounded down to 0.05
                    
        return (np.round(activity * 100 / 5, decimals=0) * 5).reshape((self.steps+2,) + (self.agent_model.grid)).tolist()
    
    def get_activties_at_step(self, position, rgba = False, comparative = False, **kwargs):

        output = {}
        for node in self.nodes:
            #in in the JS plugins an integer percentage value is used because decimals in javascript are strange hence * 100
            max_activity = (node.storage if node.storage else 1) / 100        
            mask = node.perturbed_activities
            arr = node.activities
            activity = np.where(mask[step] & (arr[step] == 0), 1, np.where(mask[step], arr[step], 0))

            if comparative:
                mask = node.control_perturbed_activities
                arr = node.control_activities
                control_activity = np.where(mask[step] & (arr[step] == 0), 1, np.where(mask[step], arr[step], 0))
                activity -= control_activity
                
            output[node] = (np.round(activity / max_activity / 5, decimals=0) * 5).tolist()
        
        return output
        
        # return {"sa" + str(x) + "_" + str(y): [node.activity_color(step_number = step, pos = (x,y), rgba = rgba) for step in range(self.steps)] for x in range(node.model.grid[0]) for y in range(node.model.grid[1])}
        # return {str(i) + "_" + str(j): arr[i, j] for i in range(arr.shape[0]) for j in range(arr.shape[1])}

    def show_submap(self, position = (0,0), submap = "", **kwargs):

        if submap in self.agent_model.files:
            model = self.agent_model
        elif submap in self.outside_model.files:
            model = self.outside_model
        else:
            "File is not a submap."
            return
            
        model.show_boolean_simulation(submap, min_steps = 0, max_steps = self.steps, slider_step = 1, pos = position, prevent_run = True, **kwargs)

    def show_activity_plot(self, nodes, normalize = False, pi = 50, steps = None):
        plt.title("Node Activities")
        plt.xlabel("Simulation Step")
        plt.ylabel("Activity Values")        
        if steps == None:
            steps = self.steps
        
        non_empty_agents = (self.agents != 0).flatten()
            
        for node in nodes:
            
            activity, perturbation = self.agent_model.restore_matrix_at_node(node)
            if node.storage:
                activity /= node.storage
            activity = np.where(perturbation == 1, 1, activity)
            if len(node.refill_sources) > 0:
                for refill_node in node.refill_sources:
                    refill_activity, refill_perturbation = self.agent_model.restore_matrix_at_node(refill_node)
                    activity = np.where(((refill_activity > 0) & (refill_perturbation != -1)) | (refill_perturbation == 1), 1, activity)   
            activity = np.where(perturbation == -1, 0, activity)
            if node.storage and not normalize:
                activity *= node.storage
                
            x, y = [], []

            for step in range(steps):
                # Apply non-zero mask directly when appending to x and y
                x.extend([step] * np.count_nonzero(non_empty_agents))
                y.extend(activity[step].flatten()[non_empty_agents])
            
            sns.lineplot(x=x, y=y, errorbar = ("pi", pi), label=node.fullname_printable)
        
        plt.legend(loc= "upper left")
        plt.show()
        

    def get_pi_activities(self, nodes, window_size = 0, normalize = False, pi = 50, steps = None):
    
        if steps == None:
            steps = self.steps
        
        non_empty_agents = (self.agents != 0).flatten()
        
        data = {
            "steps": steps,
            "activities":{},
       }
            
        for node in nodes:
            
            data["activities"][node.hash] = []
            
            activity, perturbation = self.agent_model.restore_matrix_at_node(node)
            if node.storage:
                activity /= node.storage
            activity = np.where(perturbation == 1, 1, activity)
            if len(node.refill_sources) > 0:
                for refill_node in node.refill_sources:
                    refill_activity, refill_perturbation = self.agent_model.restore_matrix_at_node(refill_node)
                    activity = np.where(((refill_activity > 0) & (refill_perturbation != -1)) | (refill_perturbation == 1), 1, activity)   
            activity = np.where(perturbation == -1, 0, activity)
            if node.storage and not normalize:
                activity *= node.storage
            
            if not node.storage and window_size > 1:
                cum_sum = np.cumsum(np.vstack([np.zeros(activity.shape[1]), activity]), axis=0)
                moving_sums = np.array([cum_sum[min(i + window_size, cum_sum.shape[0] - 1), :] - cum_sum[i, :] 
                                        for i in range(activity.shape[0])])
                window_sizes = np.array([window_size if i + window_size <= activity.shape[0] else activity.shape[0] - i for i in range(activity.shape[0])]).reshape(-1, 1)
                activity = moving_sums / window_sizes

            for step in range(steps):
                # Apply non-zero mask directly when appending to x and y
                step_activities = activity[step].flatten()[non_empty_agents]
                data["activities"][node.hash].append([
                    np.percentile(step_activities, int(50-pi/2)),
                    np.median(step_activities),
                    np.percentile(step_activities, int(50+pi/2))
                ])
            
        return data
