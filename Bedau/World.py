import typing as tp
import numpy as np
Location = tp.Tuple[int, int]


class World():
    def __init__(self, world_size, random_source):
        self.world_size = world_size
        self.world = np.zeros((world_size, world_size), dtype=float)
        self.random_source = random_source
        self.peak_resource = 2*255
        self.length=16 #width of the pyramid, 8 on each side
        self.aploc=[] #list of appropiate locations where pyramid can go
        self.bin_size = self.peak_resource / 4
        self.find_loc() #create aploc array
        self.random_source.shuffle(self.aploc) #create order to follow to prevent overlap
        # add some resource pyramids to initialize the world
        for i in range(20):
            self.generate_resources(self.aploc[i])
        self.aploc = self.aploc[20:]+self.aploc[:20]
        self.consumption_counter = 0.

    def find_loc(self):
        y=0
        x=int((self.length-1)/2)
        z=int((self.length-1)/2)
        for j in range(int(self.world_size/self.length)):
            for i in range(int(self.world_size/self.length)):
                self.aploc.append((x,z+y))
                y+=self.length
            x+=self.length
            y=0

    def generate_resources(self, loc: Location):
        #creates pyramid
        for dx in range(-8, 8):
            for dy in range(-8, 8):
                distance = abs(dx) + abs(dy)
                if distance < 8:
                    resource = 2*(255 * (1 - distance / 8))
                else:
                    resource = 0.
                self.set((loc[0] + dx, loc[1] + dy), resource)

    def remove_resources(self, loc: Location):
        #removes pyramid
        for dx in range(-8, 8):
            for dy in range(-8, 8):
                distance = abs(dx) + abs(dy)
                if distance < 8:
                    resource = 0
                else:
                    resource = 0.
                self.remove((loc[0] + dx, loc[1] + dy), resource)

    def generate_inv_resources(self,loc: Location):
        #create inverted pyramid
        for ax in range(-8,8):
            for ay in range(-8,8):
                r=126
                self.set((loc[0] + ax, loc[1] + ay), r)
        for dx in range(-8, 8):
            for dy in range(-8, 8):
                distance = abs(dx) + abs(dy)
                if distance < 8:
                    resource = -(126 * (1 - distance / 8))
                else:
                    resource = 0.
                self.set((loc[0] + dx, loc[1] + dy), resource)

    def sense(self, loc: Location):

        resources = self.get(loc)
        if(resources >= self.peak_resource):
            return 3
        else:
            return int(resources / self.bin_size)

    def get_sensory_state(self, loc: Location):

        partial_state = []
        partial_state.append(self.sense(loc))
        partial_state.append(self.sense((loc[0] + 1, loc[1])) << 2)
        partial_state.append(self.sense((loc[0] - 1, loc[1])) << 4)
        partial_state.append(self.sense((loc[0], loc[1] - 1)) << 6)
        partial_state.append(self.sense((loc[0], loc[1] + 1)) << 8)

        return sum(partial_state)

    def get(self, loc: Location):

        return self.world[loc[0] % self.world_size][loc[1] % self.world_size]

    def set(self, loc: Location, increment: float):

        self.world[loc[0] % self.world_size][loc[1] %
                                                 self.world_size] += increment

    def remove(self, loc: Location, increment: float):

        self.world[loc[0] % self.world_size][loc[1] %
                                                 self.world_size] = increment

    def probe(self, loc):
        resource_available = self.get(loc)
        resource_to_collect = min(resource_available, 100)
        self.set(loc, -resource_to_collect)
        return resource_to_collect

    def random_loc_agent(self):
        return (self.random_source.choice(self.world_size),
                self.random_source.choice(self.world_size))

    def residual_resource(self):
        return self.world.sum()
