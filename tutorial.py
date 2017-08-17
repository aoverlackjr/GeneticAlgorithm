from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt

# Demonstration script to show functionality of genetic algorithm
# The example problem we are trying to solve is a simple mathematical parametric equation.
# The equation is the following;

# A*x^e1 + B*x^e2 = pi
x = 10
correct_answer = 3.14159
# Where x is a known number (10) and A,B and exponents e1 and e2 are unknown. Of course, there are more trivial ways of
# finding a solution to this problem, and multiple solutions exist.
# The way we proceed with a genetic algorithm is as follows:

# We define a population of 20 individuals per generation, where each individual
# has a gene length of 4. This is because we will encode the individual (solution) according to four parameters:
# A, B, e1 and e2.
# We will use a float as the definition of a gene.
popsize = 20
GA = GeneticAlgorithm(  crossover_rate = 0.7,
                        mutation_rate = 0.001,
                        pop_size = popsize,
                        gene_length = 4,
                        nr_of_genes = 75,
                        max_generations = 400,
                        max_mutation_perturbation = 0.3,
                        elitism = False,
                        mode = 'float',
                        float_bias = 0.0,
                        float_range = 1.5)

# Now, for some arbitrary amount of times we allow a generations to attack the problem.
nr_of_generations = 200

# THIS WOULD BE THE BODY OF THE MAIN (MASTER) APPLICATION
for generationNr in range(nr_of_generations):
    # For each generation we assess each individual:
    print "Running generation: %(gennr)d" % {"gennr": generationNr}
    for individualNr in range(popsize):
        # The individuals genes are used to approach the problem, we directly 'de-code' the gene by using the floats as-is
        # from the chromosome of the selected individual;
        chromo = GA.individual(individualNr)
        A   = chromo[0]
        B   = chromo[1]
        e1  = chromo[2]
        e2  = chromo[3]
        result = A*pow(x,e1) + B*pow(x,e2)
        print result
        # Now, the result is taken to see if it is a 'fit' solution. There are many ways to
        # define the fitness numerically. In some cases the right answer is not available, so some other metric needs to be used.
        # It is advised to use a number between 0 and 1 (normalized), as large (and highly non-linear) fitness
        # assignments will lead to convergence problems.
        # Here we will assign a number between 0 and 1 in the following way:
        diff_sq = pow(abs(result - correct_answer) + 1.0, 2.0)
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
A   = chromo[0]
B   = chromo[1]
e1  = chromo[2]
e2  = chromo[3]
result_fittest = A*pow(x,e1) + B*pow(x,e2)
print "The fittest individual obtains a result of: %(res)f and was found in generation: %(gennr)d at index: %(ind)d and has a fitness of: %(fit)f" % {"gennr": generationNr, "res":result_fittest, "ind":indexNr, "fit":fitness}

# Varying the input argument to the GA constructor will show different convergence bahaviours.
# Repeated runs with similar parameters will also show that sometimes an acceptable solution is never found
