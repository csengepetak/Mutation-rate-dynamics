from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation
from datetime import datetime as dt
import numpy as np
import csv
csvfile = "static/results_base.csv"
csvfile2 = "static/results_score.csv"
from collections import Counter

class Log():
    def __init__(self, iterations, plot_world_flag=False):
        self.iterations = iterations
        self.plot_world_flag = plot_world_flag
        self.track_world = []
        self.track_population = []
        self.track_residual = []
        self.track_color = []
        self.track_other_color = []
        self.track_mutations = []
        self.final_pop = [] #sensory_motor_map of all agents in the end
        self.sensory_list=list(range(0,1024)) #sensory states in the map
        self.mosties=[] #most common solutions - behaviours in positions
        self.score=[] #how many positions were common in the pop
        self.final_mutations=[] #mutation rates of agents in the end


    def log_stats(self, world, population,color, other_color, iteration):
        residual = world.residual_resource()
        self.track_residual.append(residual)
        pop_size = len(population)
        self.track_population.append(pop_size)

        mean_mut = 0.
        temp = []
        color = 0
        other_color = 0

        valid=list(range(100,20000,99)) #save score at every 1000th iteration
        if iteration in valid:
            #set to empty at every "checkpoint" iteration
            self.final_pop=[]
            self.final_mutations=[]
            for agent in population:
                self.final_pop.append(agent.sensory_motor_map)
                self.final_mutations.append(agent.mutation_rate)
            #self.write_csv_score()

        if pop_size > 0:
            for agent in population:
                mean_mut += agent.mutation_rate
                if self.plot_world_flag: #for animation
                    temp.append((agent.position[1], agent.position[0]))
            mean_mut /= pop_size

        for agent in population:
            if agent.color == 1:
                color += 1
            else:
                other_color += 1
        self.track_color.append(color)
        self.track_other_color.append(other_color)
        self.track_mutations.append(mean_mut)

        if self.plot_world_flag:
            locations = np.array(temp)
            self.track_world.append((np.array(world.world), locations))

        return residual, pop_size, color, other_color, mean_mut

    def plot_world(self):
        fig, ax = plt.subplots()
        ax = plt.axes(xlim=(0, 128), ylim=(0, 128))
        line1 = ax.imshow(self.track_world[0][0], shape=(128, 128),
                          interpolation='nearest', cmap=cm.coolwarm)
        line2 = ax.scatter([], [], s=10, c='red')

        def init():
            line1.set_array([[], []])
            line2.set_offsets([])
            return line1, line2

        def animate(i):
            line1.set_array(self.track_world[i][0])
            line2.set_offsets(self.track_world[i][1])
            return line1, line2

        anim = animation.FuncAnimation(fig, animate, frames=len(
            self.track_world), interval=300, blit=True, init_func=init, repeat=False)
        time = dt.now()
        path_to_save = time.strftime('%Y-%m-%d_%H-%M') + '.mp4'
        print('Plotting track_world to ' + path_to_save)
        anim.save(path_to_save, fps=5, dpi=300,
                  extra_args=['-vcodec', 'libx264'])
        print('Plotting Finished')

    def most_solution(self):
        self.sum_sensoryval()
        self.mosties=[]
        self.score=[]
        a=np.array(self.final_pop)
        rows,columns = a.shape
        for column in range(columns):
            n,m =Counter(a[:,column]).most_common(1)[0] #gives back most common with its frequency per position
            new_val = int(n/8) + (n%8>0) #count number of steps from behaviour
            self.mosties.append((new_val,n,m))
        self.mosties=[ (val, be,count, sense) for (val,be,count),sense in zip(self.mosties,self.sensory_list)] #put in one with sensory map

        score=0
        for i in range(1024):
            if self.mosties[i][2]/len(self.final_mutations) > 0.5:
                score+=1
        self.score.append(score)
        return self.mosties

    def write_csv_all(self):
        self.most_solution()
        with open(csvfile, "w") as output: #overwrites file
            writer = csv.writer(output, delimiter=';',lineterminator='\n')
            writer.writerow(["Steps", "Behaviour", "Frequency","State"])
            for val in self.mosties:
                writer.writerow(val)

    def write_csv_score(self):
        self.most_solution()
        with open(csvfile2, "a") as output: #append to existing file, doesn't overwrite
            writer = csv.writer(output, delimiter=';',lineterminator='\n')
            writer.writerow(self.score)

    def decode_sensory(self,state):
        #from sensory_motor_map position gives back [0,1,1,2,3] where these are the bin value in order:
        #right, left, up, down, middle
        values = []
        shift=0
        and_=3
        for i in range (5):
            temp = (state & and_)>>shift
            and_*=4
            shift+=2
            values.insert(0,temp)
        return values

    def sum_sensoryval(self): #to give amout of food in position
        for i in range(0,1024):
            x=self.decode_sensory(i)
            #y=sum(x)
            self.sensory_list[i]=x

    def plot_stats(self, save_stats=True):
        x = np.arange(self.iterations)
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(x, self.track_residual)
        axarr[0].set_ylabel('Residual resources')
        axarr[1].plot(x, (self.track_other_color), 'b')
        axarr[1].plot(x, self.track_color, 'r')
        axarr[1].set_ylabel("Number of color 1's")
        axarr[2].plot(x, self.track_mutations)
        axarr[2].set_ylabel('Avg mut. rate')
        axarr[2].set_xlabel('Iterations')

        for ax in axarr:
            # ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # set vertical dashed lines to better check values across plots
            ax.grid(True, axis='x', linestyle='dashed')
            ax.get_yaxis().set_label_coords(-0.17, 0.5)

        # show x axis values only on last plot
        plt.setp([a.get_xticklabels() for a in axarr[:-1]], visible=False)
        if save_stats:
            time = dt.now()
            path_to_save = time.strftime('%Y-%m-%d_%H-%M') + '.pdf'
            print('Plotting stats to ' + path_to_save)
            plt.savefig(path_to_save, bbox_inches='tight')
            print('Plotting Finished')
        else:
            plt.tight_layout()
            plt.show()
