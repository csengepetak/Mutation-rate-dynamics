from Bedau.World import World
from Bedau.Agent import Agent
from Bedau.Log import Log
import numpy as np
import math
import random


class Population():
    def __init__(self, world_size, pop_size, mutation_rate, meta_mutation, meta_mutation_range, iterations, plotting=False, seed=None, progress=False):
        print("Generating population")
        self.progress = progress
        self.world_size = world_size
        # mutation rate: μ
        self.mutation_rate = mutation_rate
        # mutation rate of mutation: μ_μ
        self.meta_mutation = meta_mutation
        # mutation rate varies in [μ-ε,μ+ε]: ε
        self.meta_mutation_range = meta_mutation_range
        self.loclist=list(range(0,100000))
        #locations where the pyramids get generated to keep track and remove later
        self.behaviours = [(0, 0, 0)]
        coeff = math.sqrt(2)
        for inc in range(1, 16):
            self.behaviours.append((inc, 0, inc))
            self.behaviours.append((-inc, 0, inc))
            self.behaviours.append((0, inc, inc))
            self.behaviours.append((0, -inc, inc))
            magnitude = coeff * inc
            self.behaviours.append((inc, inc, magnitude))
            self.behaviours.append((-inc, -inc, magnitude))
            self.behaviours.append((inc, -inc, magnitude))
            self.behaviours.append((-inc, inc, magnitude))
        # possible behaviours are 1-15 steps in 8 compass direction + (0,0)
        #9 behaviours in order: stay, South, North, East, West, SE, NW, SW, NE
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.SystemRandom().getrandbits(32)
        #for if we want to repeat a specific run
        self.random_source = np.random.RandomState(self.seed)
        self.world = World(world_size, self.random_source)
        self.population = []
        self.iterations = iterations
        self.log = Log(self.iterations, plot_world_flag=plotting)

        for i in range(0, pop_size):
            self.population.append(
                Agent(world_size=self.world_size,
                      behaviours=self.behaviours,
                      mutation_parameters=(self.mutation_rate, self.meta_mutation, self.meta_mutation_range),
                      world=self.world,
                      random_source=self.random_source,
                      log=self.log))
        print("Population ready")

    def evolve(self):
        for idx in range(self.iterations):

            count=(idx)%len(self.world.aploc)
            #world.aploc is an array of locations where the pyramid can be
            #generated, randomly shuffled for every run. once all aploc
            #are used up it goes back to the begining. it is 64 long, so removal
            #of old pyramid should happen before strating the cycle again.
            if idx < 60:
                self.loclist[idx]=self.world.aploc[count]
                self.world.generate_resources(self.loclist[idx])
            else:
                self.loclist[idx]=self.world.aploc[count]
                self.world.generate_resources(self.loclist[idx])
                #from the given iterations it changes the format of the pyramids
                #and starts removing old ones. this can be separated later
                self.world.remove_resources(self.loclist[idx-60])

            self.update_pop(idx)
            self.stats(idx)
            if len(self.population) == 0:
                self.log.iterations = idx+1
                break
        print("\nSeed: {}\n".format(self.seed))
        return self.log

    def update_pop(self, iteration):
        new_pop = []
        # shuffle to avoid giving proiority to any specific agent during update
        self.random_source.shuffle(self.population)
        for agent in self.population:
            state, child = agent.update(iteration)
            if(state == True):
                new_pop.append(agent)
                if(child is not None):
                    new_pop.append(child)
        self.population = new_pop

    def stats(self, iteration):
        residual, pop_size, color, other_color, mean_mut = self.log.log_stats(
            self.world, self.population, self.population, self.population, iteration)
        if self.progress:
            print("---------------------")
            print("Iteration: {}".format(iteration))
            print("Pop size: {}".format(pop_size))
            print("Residual resource: {}".format(residual))
            print("Color: {}".format(color))
            print("Mean mutation: {}".format(mean_mut))
