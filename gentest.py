<<<<<<< HEAD
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
import cb


vec = np.array([2,3,5,7,11,13,17])
ans = np.array([2,4,6,8])

nv = len(vec)
na = len(ans)

popsize = 100
GA = GeneticAlgorithm(  crossover_rate              = 0.5,
                        mutation_rate               = 0.01,
                        pop_size                    = popsize,
                        gene_length                 = 1,
                        nr_of_genes                 = nv*na,
                        max_generations             = 400,
                        sigma_mutation_perturbation = 0.01,
                        elitism                     = True,
                        mode                        = 'float',
                        float_bias                  = 0.0,
                        float_range                 = 0.5,
                        nr_of_crossovers            = 3,
                        crossover_mode              = 'uniform',
                        progenitor                  = None,
                        roulette_mode               = 'rank')

# Now, for some arbitrary amount of times we allow a generations to attack the problem.
nr_of_generations = 300

# THIS WOULD BE THE BODY OF THE MAIN (MASTER) APPLICATION
print('Evolving...\n')
for generationNr in range(nr_of_generations):
    # For each generation we assess each individual:
    #print("Running generation: {}\n".format(generationNr))
    for chromo, individualNr in GA.individuals():
        # The individuals genes are used to approach the problem, we directly 'de-code' the gene by using the floats as-is
        # from the chromosome of the selected individual;
        # chromo = GA.individual(individualNr)
        matrix = np.array(chromo).reshape(na,nv)
        
        result = matrix.dot(vec)
        #print(result)
        # Now, the result is taken to see if it is a 'fit' solution. There are many ways to
        # define the fitness numerically. In some cases the right answer is not available, so some other metric needs to be used.
        # It is advised to use a number between 0 and 1 (normalized), as large (and highly non-linear) fitness
        # assignments will lead to convergence problems.
        # Here we will assign a number between 0 and 1 in the following way:
        diff_sq = pow(np.linalg.norm(result-ans) + 1.0, 2.0)
        fitness_of_this_one = 1.0/(diff_sq)
        GA.assignFitness(fitness_of_this_one, individualNr)
    # We have assed one generation, and now we need to cycle it:
    GA.cycle()
# END OF MAIN APPLICATION

# See how our generation evolved over time:
average_fitnesses, max_fitnesses = GA.getFitnessEvolution()
plt.plot(average_fitnesses)
plt.plot(max_fitnesses)
plt.ylabel('Average and max fitnesses')
plt.show()

# See what the fittest individual in history does:
chromo, fitness, generationNr, indexNr = GA.getFittestFromEvolution()
matrix = np.array(chromo).reshape(na,nv)
result_fittest = matrix.dot(vec)
print("The fittest individual obtains a result of: {} and was found in generation: {} at index: {} and has a fitness of: {}\n".format(result_fittest, generationNr, indexNr, fitness))

GA.save_chromo_to_file(chromo, 'best_one_yet', 'cb.py', overwrite = True)
# Varying the input argument to the GA constructor will show different convergence behaviours.
# Repeated runs with similar parameters will also show that sometimes an acceptable solution is never found
print(matrix)

=======
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np


vec = np.array([2,3,5,7,11,13])
ans = np.array([2,4,6,8])

nv = len(vec)
na = len(ans)


popsize = 50
GA = GeneticAlgorithm(  crossover_rate              = 0.7,
                        mutation_rate               = 0.01,
                        pop_size                    = popsize,
                        gene_length                 = 1,
                        nr_of_genes                 = nv*na,
                        max_generations             = 400,
                        sigma_mutation_perturbation   = 0.01,
                        elitism                     = True,
                        mode                        = 'float',
                        float_bias                  = 0.0,
                        float_range                 = 0.1,
                        nr_of_crossovers            = 15,
                        crossover_mode              = 'fixed',
                        progenitor                  = None)

# Now, for some arbitrary amount of times we allow a generations to attack the problem.
nr_of_generations = 500

# THIS WOULD BE THE BODY OF THE MAIN (MASTER) APPLICATION
print('Evolving...\n')
for generationNr in range(nr_of_generations):
    # For each generation we assess each individual:
    #print("Running generation: {}\n".format(generationNr))
    for chromo, individualNr in GA.individuals():
        # The individuals genes are used to approach the problem, we directly 'de-code' the gene by using the floats as-is
        # from the chromosome of the selected individual;
        # chromo = GA.individual(individualNr)
        matrix = np.array(chromo).reshape(na,nv)
        
        result = matrix.dot(vec)
        #print(result)
        # Now, the result is taken to see if it is a 'fit' solution. There are many ways to
        # define the fitness numerically. In some cases the right answer is not available, so some other metric needs to be used.
        # It is advised to use a number between 0 and 1 (normalized), as large (and highly non-linear) fitness
        # assignments will lead to convergence problems.
        # Here we will assign a number between 0 and 1 in the following way:
        diff_sq = pow(np.linalg.norm(result-ans) + 1.0, 2.0)
        fitness_of_this_one = 1.0/(diff_sq)
        GA.assignFitness(fitness_of_this_one, individualNr)
    # We have assed one generation, and now we need to cycle it:
    GA.cycle()
# END OF MAIN APPLICATION

# See how our generation evolved over time:
average_fitnesses, max_fitnesses = GA.getFitnessEvolution()
plt.plot(average_fitnesses)
plt.plot(max_fitnesses)
plt.ylabel('Average and max fitnesses')
plt.show()

# See what the fittest individual in history does:
chromo, fitness, generationNr, indexNr = GA.getFittestFromEvolution()
matrix = np.array(chromo).reshape(na,nv)
result_fittest = matrix.dot(vec)
print("The fittest individual obtains a result of: {} and was found in generation: {} at index: {} and has a fitness of: {}".format(result_fittest, generationNr, indexNr, fitness))

GA.save_chromo_to_file(chromo, 'best_one_yet', 'cb.py', overwrite = True)
# Varying the input argument to the GA constructor will show different convergence behaviours.
# Repeated runs with similar parameters will also show that sometimes an acceptable solution is never found
>>>>>>> a3d8606fb791212c4e4280322c68704bd3e54369
