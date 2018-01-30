from Bedau.Population import Population
from Bedau.Log import Log


def main():
    print("Start")
    world_size = 128
    pop_size = 1000
    mutation_rate = 0.01
    meta_mutation = 0.66
    meta_mutation_range = 0.01
    plotting = True #video
    iterations = 100
    pop_log = Population(world_size=world_size,
                         pop_size=pop_size,
                         mutation_rate=mutation_rate,
                         meta_mutation=meta_mutation,
                         meta_mutation_range=meta_mutation_range,
                         iterations=iterations,
                         plotting=plotting,
                         progress=True).evolve()

    if plotting:
        pop_log.plot_world() #video
    pop_log.plot_stats() #graph
    #pop_log.write_csv_all() # most common behaviours at positions in the end
    #pop_log.write_csv_score() # at intermediate steps how many positions that are
                            #popular in the population (above 50%)
    print("End")

if __name__ == '__main__':
    for i in range(1):
        main()
