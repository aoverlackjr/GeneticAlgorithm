import numpy as np
import os
from copy import deepcopy as clone
from random import sample
from math import sqrt, ceil

class GeneticAlgorithm(object):
    '''
    Genetic Algorithm object to support the housekeeping of any genetic algorithm based
    optimization or simulation. It uses the basic implementations of a genetic algorithm.
    Some more advanced functions are still to be implemented.
    The object itself supports the main (specific) application. The interface to this object,
    and the associated workflow is to be implemented in the main algorithm. The high level workflow
    is the following:
    1) This object is created with the relevant parameters.
    2) A first population of (random) individuals is created.
    3) The population is basically a list of genes, that can be accessed externally.
    4) In the main application, these genes are interpreted in the application-specific way. (Gene-to-modelparameters)
    5) In the main application a fitness number is assigned after simulation/assesment of the individual's performance.
    6) The main application calls this object's cycle function after all individuals have been assessed, leading to a new (non-random) population.
    7) Step 4 and 5 are repeated until an individual is found that meets certain criteria.
    '''
    def __init__(self,
                 crossover_rate = 0.7,                  # The probablity gene cross-over occurs during mating. (0-1)
                 mutation_rate = 0.001,                 # The pobablity of mutations occuring. (0-1)
                 pop_size = 100,                        # The size of the population. (Nr. of indiv.)
                 gene_length = 4,                       # The length of the gene. If digital, this is the number of bits encoding one gene.
                 nr_of_genes = 75,                      # The number of genes in an individuals chromosome.
                 max_generations = 400,                 # A placeholder cap of simulation duration (or: cycles). Does not have to be used.
                 sigma_mutation_perturbation = 0.3,     # In case the gene is encoded as a float, the mutation is also a float with a maximum perturbation.
                 elitism = True,                        # Elitism enables the fittest individual to go through to the next generation unaltered.
                 mode = 'digital',                      # digital / float encoding of the gene. Choice is application specific.
                 float_bias = 0.0,                      # The starting bias of the floats in the gene.
                 float_range = 1.0,                     # The starting range of the float size.
                 nr_of_crossovers = 1,                  # The amount of crossover points when crossing over.
                 crossover_mode = 'fixed',              # Whether fixed or uniform crossover is used
                 progenitor = None,                     # The seed chromo, if any
                 roulette_mode = 'fitness',             # The mode for selecting the fittest individuals
                 random_distribution = 'normal'):

        self.crossover_rate              = crossover_rate
        self.mutation_rate               = mutation_rate
        self.pop_size                    = pop_size
        self.gene_length                 = gene_length
        self.nr_of_genes                 = nr_of_genes
        self.max_generations             = max_generations
        self.sigma_mutation_perturbation = sigma_mutation_perturbation
        self.elitism                     = elitism
        self.float_range                 = float_range
        self.float_bias                  = float_bias
        self.nr_of_crossovers            = nr_of_crossovers
        self.f_crossOver                 = self._crossOver
        self.f_roulette                  = self._fitness_roulette
        self.random_distribution         = random_distribution

        self.mean_diversity = []
        self.mean_deviation = []

        # If the mode is set to float, a single nr is a gene.
        if mode == 'float' and gene_length > 1:
            self.gene_length = 1

        # Safeguard for wrong input.
        if mode != 'float' and mode != 'digital':
            mode = 'digital'

        # set crossover function
        if crossover_mode == 'fixed':
            self.f_crossOver = self._crossOver
        if crossover_mode == 'uniform':
            self.f_crossOver = self._crossOverUniform

        self.nr_of_bits = self.gene_length*self.nr_of_genes

        if progenitor:
            if self.nr_of_bits == len(progenitor):
                self.progenitor = progenitor
            else:
                self.progenitor = None
                print('Progenitor not of compatible length, not seeding...')
        else:
            self.progenitor = None

        roulette_modes = {'fitness' : self._fitness_roulette,
                          'rank'    : self._rank_roulette}

        self.f_roulette = roulette_modes[roulette_mode]

        # Set the mode, generation counter and init first population.
        self.mode                      = mode
        self.generation_nr = 0
        self._createPopulation()

    def cycle(self):
        # The cycle function is called externally when a population has been suitably assesed for fitness.
        # It takes the current population and their (externally) assigned fitness to create
        # a new population.
        # Create a new population;
        
        # keep track of the diversity of the current population:
        m, sigma = self._get_diversity()
        self.mean_diversity.append(m)
        self.mean_deviation.append(sigma)
        
        newGeneration = Generation(self.pop_size)
        # Select pairs in the population and mate them.
        for ind in range(0,self.pop_size,2):
            chromo1, index1 = self._roulette()
            chromo2, index2 = self._roulette()
            # If the same individual is chosen twice, perform this step until
            # two different ones are found, and also prevent incest
            if self._is_same_genotype(chromo1, chromo2) or self._are_of_same_parents(index1, index2):
                # If the set of chromos is identical, or if parents are shared, loop until a set is found
                # which satisfies a healthy condition
                while self._is_same_genotype(chromo1, chromo2) and self._are_of_same_parents(index1, index2):
                    chromo1, index1 = self._roulette()
                    chromo2, index2 = self._roulette()

            offspring1, offspring2 = self._mate(chromo1,chromo2)
            newGeneration.individuals.append(offspring1)
            newGeneration.individuals.append(offspring2)
            # both chromos have the same parents
            newGeneration.lineage.append((index1, index2))
            newGeneration.lineage.append((index1, index2))
        # If elitism is true, the fittest pair of individuals go through to the next
        # generation at a random index of the population.
        if self.elitism:
            elite1, elite2 = self._findFittestPair()
            indices = sample(range(0,self.pop_size),2)
            newGeneration.individuals[indices[0]] = elite1
            newGeneration.individuals[indices[1]] = elite2
        self.history.append(newGeneration)
        self.generation_nr += 1

    def individual(self, index):
        # Get the chromosome of the individual of the current population
        return self.history[self.generation_nr].individuals[index]

    def assignFitness(self, fitness, index):
        # Assign fitness to individual at the index of the current population.
        self.history[self.generation_nr].fitness[index] = fitness

    def getFittestFromEvolution(self):
        # At a certain moment the user may be interested to find the fittest
        # individual in the entire history of the evolution, as it may not be an
        # individual in the last generation (population)
        max_fitness = -1.0
        gen = 0
        for generation in self.history:
            maxfit = max(generation.fitness)
            ind = generation.fitness.index(maxfit)
            if maxfit > max_fitness:
                max_fitness = maxfit
                generation_index = gen
                individual_index = ind
            gen += 1
        chromo = clone(self.history[generation_index].individuals[individual_index])
        fitness = self.history[generation_index].fitness[individual_index]
        return chromo, fitness, generation_index, individual_index

    def getFitnessEvolution(self):
        # A helper function to help in plotting the evolution of the fitness.
        # This can be used to see if the genetic approach is converging on an (optimal)
        # solution, or whether solutions (individuals) are getting worse.
        av_fitness = []
        max_fitness = []
        for generation in self.history:
            av_fitness.append(sum(generation.fitness)/float(len(generation.fitness)))
            max_fitness.append(max(generation.fitness))
        return av_fitness, max_fitness

    def _createPopulation(self):
        # Function to initialize the objects population data and history.
        self.history = []
        generation = Generation(self.pop_size)
        if self.mode == 'digital':
            for individualNr in range(self.pop_size):
                generation.individuals.append(self._generateRandomByte(self.gene_length * self.nr_of_genes))
        elif self.mode == 'float':
            for individualNr in range(self.pop_size):
                generation.individuals.append(self._generateRandomFloats(self.nr_of_genes, self.float_range, self.float_bias))
        # add seed, if any
        if self.progenitor:
            generation.individuals[0] = self.progenitor
        self.history.append(generation)

    def _crossOver(self, chromo1, chromo2):
        # The cross-over function takes two individuals (chromosomes) and mates them
        # according to a stochastical index.
        offspring1 = clone(chromo1)
        offspring2 = clone(chromo2)
        if np.random.rand() < self.crossover_rate:
            random_indices, switches = self._generateRandomIndices(self.nr_of_crossovers, self.nr_of_bits)
            for i in range(self.nr_of_bits):
                if switches[i]:
                    offspring1[i] = chromo2[i]
                    offspring2[i] = chromo1[i]
        return offspring1, offspring2

    def _crossOverUniform(self, chromo1, chromo2):
        offspring1 = clone(chromo1)
        offspring2 = clone(chromo2)
        for i in range(self.nr_of_bits):
            if np.random.rand() < self.crossover_rate:
                offspring1[i] = chromo2[i]
                offspring2[i] = chromo1[i]
        return offspring1, offspring2

    def _mutateChromosome(self, chromo_in):
        # The mutation function implements (random) mutations
        chromo_out = clone(chromo_in)
        if self.mode == 'digital':
            for i in range(len(chromo_in)):
                if np.random.rand() < self.mutation_rate:
                    if chromo_in[i] == 1:
                        chromo_out[i] = 0
                    elif chromo_in[i] == 0:
                        chromo_out[i] = 1
        elif self.mode == 'float':
            for i in range(len(chromo_in)):
                if np.random.rand() < self.mutation_rate:
                    chromo_out[i] = chromo_in[i] + np.random.randn()*self.sigma_mutation_perturbation
        return chromo_out

    def _mate(self,chromo1,chromo2):
        # The mate function wraps the cross-over and mutation actions.
        offspring1, offspring2 = self.f_crossOver(chromo1,chromo2)
        offspring1 = self._mutateChromosome(offspring1)
        offspring2 = self._mutateChromosome(offspring2)
        return offspring1, offspring2

    def _findFittestPair(self):
        # Find the fittest pair, helper function for elitism.
        fitness_copy = clone(self.history[self.generation_nr].fitness)
        index1 = fitness_copy.index(max(fitness_copy))
        fitness_copy[index1] = 0.0
        index2 = fitness_copy.index(max(fitness_copy))
        return clone(self.history[self.generation_nr].individuals[index1]), clone(self.history[self.generation_nr].individuals[index2])

    def _roulette(self):
        chromo, index = self.f_roulette()
        return chromo, index

    def _rank_roulette(self):
        sc,sf,si = self.history[self.generation_nr].sort_by_rank()
        total_rank = 0.5*(self.pop_size**2 + self.pop_size)
        slider = np.random.rand()*total_rank
        index = int(ceil((-1.0 + sqrt(1.0 + 8.0*slider))/2.0) - 1)
        chromo = sc[index]
        return chromo, si[index]

    def _fitness_roulette(self):
        # A function to select individuals in a generation fro the mating process.
        # The selection is done according to the fitness and a stochastic parameter.
        total_fitness = sum(self.history[self.generation_nr].fitness)
        slider = np.random.rand()*total_fitness
        cumulative_fitness = 0.0
        found_chromo = False
        for chrNr in range(self.pop_size):
            cumulative_fitness += self.history[self.generation_nr].fitness[chrNr]
            if cumulative_fitness >= slider:
                chromo = clone(self.history[self.generation_nr].individuals[chrNr])
                found_chromo = True
                break
        if found_chromo:
            return chromo, chrNr
        else:
            raise Exception("Could not find chromosome in roulette.")

    def _get_average_chromosome(self):
        av_chromo = np.zeros(self.nr_of_genes)
        for chromo, nr in self.individuals():
            av_chromo += np.array(chromo)
        av_chromo = av_chromo/self.pop_size
        return av_chromo
        
    def _get_genetic_distances(self):
        center_genotype = self._get_average_chromosome()
        distances = []
        for chromo, nr in self.individuals():
            distances.append(np.linalg.norm(np.array(chromo) - np.array(center_genotype)))
        return distances
        
    def _get_diversity(self):
        deviations = self._get_genetic_distances()
        mean_deviation = np.mean(deviations)
        std_deviation  = np.std(deviations)
        return mean_deviation, std_deviation

    def _generateRandomFloats(self, length, max_range, bias):
        # Helper for the initialization of the first generation.
        if self.random_distribution == 'normal':
            return np.random.randn(length) * float(max_range) + float(bias)
            
        if self.random_distribution == 'uniform':
            return np.random.uniform(low = -max_range, high = max_range, size = length) + float(bias)
            

    def _generateRandomByte(self, length):
        # Helper for the initialization of the first generation.
        return np.random.randint(0,1,length)

    def _generateRandomIndices(self, nr_of_indices, max_index):
        random_indices = np.unique(np.sort(np.random.randint(1, max_index, nr_of_indices)))
        switch_array = np.array([True]*max_index)
        switch = True
        for i in range(max_index):
            if i in random_indices:
                switch = not switch
            switch_array[i] = switch
        return random_indices, switch_array

    def _is_same_genotype(self, chromo1, chromo2):
        is_same = True
        for g1, g2 in zip(chromo1, chromo2):
            if g1 != g2:
                is_same = False
                break
        return is_same

    def _are_of_same_parents(self, index1, index2):
        are_same = True
        # if one of the parents of the chosen chromos is similar
        parents1 = self.history[self.generation_nr].lineage[index1]
        parents2 = self.history[self.generation_nr].lineage[index2]
        if len(np.unique([parents1, parents2])) == 4:
            are_same = False
        return are_same

    def individuals(self):
        for individualNr in range(self.pop_size):
            yield self.individual(individualNr), individualNr

    def save_chromo_to_file(self, chromo, chromo_name, file_name, overwrite = False):
        option = 'a'
        if overwrite:
            option = 'w'

        with open(file_name, option) as f:
            f.write('\n')
            f.write(chromo_name + ' = ')
            string = '[ '
            for genenr in range(len(chromo)-1):
                string += str(chromo[genenr])
                string += ', '
            string += str(chromo[-1])
            string += ' ]'
            f.write(string+'\n')

    def save_generation_to_file(self):
        file_name = 'generation_store.py'
        fitness = self.history[self.generation_nr].fitness
        chromos = self.history[self.generation_nr].individuals

        fitstring    = []
        chromostring = []

        fitstring.append('fitness = [ ')
        for ind in range(self.pop_size-1):
            fitstring.append(str(fitness[ind]))
            fitstring.append(', ')
        fitstring.append(str(fitness[-1]))
        fitstring.append(' ]')

        chromostring.append('chromosomes = [ ')
        for chromonr in range(self.pop_size):
            chromostring.append('[ ')
            for genenr in range(self.nr_of_genes-1):
                chromostring.append(str(chromos[chromonr][genenr]))
                chromostring.append(', ')
            chromostring.append(str(chromos[chromonr][-1]))
            chromostring.append(' ]')
            if chromonr != self.pop_size - 1:
                chromostring.append(', ')
        chromostring.append(' ]')

        with open(file_name, 'w') as f:
            f.write(''.join(fitstring))
            f.write('\n\n')
            f.write(''.join(chromostring))

    def load_generation(self):
        import generation_store as gs
        self.history[-1].fitness = gs.fitness
        self.history[-1].individuals = gs.chromosomes
        self.cycle()

class Generation(object):

    def __init__(self, size):
        # A generation object is simply a collection of lists.
        # the list of individuals is a list of arrays (chromosomes). These are read externally.
        # The fitness list lists the fitnesses for the individuals. These are assigned externally.
        self.individuals = []
        self.fitness = []
        self.lineage = []
        for i in range(size):
            self.fitness.append(0.0)
            self.lineage.append((-i,-i))

    def sort_by_rank(self):
        fitness_sorted = []
        chromos_sorted = []
        indices = np.argsort(self.fitness)
        for index in indices:
            fitness_sorted.append(self.fitness[index])
            chromos_sorted.append(self.individuals[index])

        return chromos_sorted, fitness_sorted, indices
