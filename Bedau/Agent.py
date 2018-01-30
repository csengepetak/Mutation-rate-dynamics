import copy
import numpy as np


class Agent():

    def __init__(self, world_size=None, behaviours=None, mutation_parameters=None, color=None, world=None, random_source=None, log=None, orig=None):
        if orig is None:
            self.position = world.random_loc_agent()
            self.resources = 250.  # amount of resourses when initiated
            self.mutation_rate = mutation_parameters[0]
            self.meta_mutation = mutation_parameters[1]
            self.meta_mutation_range = mutation_parameters[2]
            self.random_source = random_source
            self.color = self.random_source.randint(0, 2)
            self.behaviours = behaviours
            self.sensory_motor_map = self.random_source.randint(121, size=1024, dtype='int8')
            self.log = log
            self.world = world

        else:
            # copy constructor
            self.position = orig.position
            self.resources = orig.resources
            self.mutation_rate = orig.mutation_rate
            self.meta_mutation = orig.meta_mutation
            self.meta_mutation_range = orig.meta_mutation_range
            self.color = orig.color
            self.sensory_motor_map = np.array(orig.sensory_motor_map)
            self.behaviours = orig.behaviours
            self.world = orig.world
            self.random_source = orig.random_source
            self.log = orig.log

    def move(self, position, behaviour):
        # behaviour = (dx,dy, magnitude), discard the magnitude with [:2]
        self.position = tuple(map(lambda x, y: (x + y) %
                                  self.world.world_size, position, behaviour[:2]))

    def reproduce(self, iteration):
        self.resources /= 2
        # the child is going to have half of the resources of the parent
        child = Agent(orig=self)
        # mutate the sensory_motor_map
        num_mutations = self.random_source.binomial(1024, self.mutation_rate)
        loc_mutations = self.random_source.randint(1024, size=num_mutations)
        new_behaviors = self.random_source.randint(121, size=num_mutations)
        child.sensory_motor_map[loc_mutations] = new_behaviors

        if self.random_source.rand(1) < self.mutation_rate:
            new_color = (self.color - 1)*(self.color - 1)
        else:
            new_color = self.color
        child.color = new_color

        # mutate the mutation rate
        if(self.random_source.rand(1) < self.meta_mutation):
            child.mutation_rate = self.random_source.uniform(
                max(0, self.mutation_rate - self.meta_mutation_range),
                min(1, self.mutation_rate + self.meta_mutation_range))

        return child

    def update(self, iteration):
        sensory_state = self.world.get_sensory_state(self.position)
        current_behaviour = self.sensory_motor_map[sensory_state]
        self.move(self.position, self.behaviours[current_behaviour])
        '''
        if iteration > 5000:

            n = 200
            if iteration %(2*n) >= n:
                if self.color == 1:
                    self.resources += self.world.probe(self.position) \
                        - 20 - self.behaviours[current_behaviour][2]
                else:
                    self.resources += self.world.probe(self.position)*2/3 - 20 - self.behaviours[current_behaviour][2]
            else:
                if self.color == 0:
                    self.resources += self.world.probe(self.position) \
                        - 20 - self.behaviours[current_behaviour][2]
                else:
                    self.resources += self.world.probe(self.position)*2/3 - 20 - self.behaviours[current_behaviour][2]
        else:
            self.resources += self.world.probe(self.position) \
                - 20 - self.behaviours[current_behaviour][2]
        '''
        self.resources += self.world.probe(self.position) \
            - 20 - self.behaviours[current_behaviour][2]

        if self.resources <= 0:
            return False, None
        if self.resources >= 500:
            return True, self.reproduce(iteration)
        else:
            return True, None
