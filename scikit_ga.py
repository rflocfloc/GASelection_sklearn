# Deap modules
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from multiprocessing.dummy import Pool

import matplotlib.pyplot as plt

# Data science modules
import numpy as np

# system modules
import random


# sklearn modules
from sklearn.model_selection import cross_val_score




##########################

class GASelection:
    def __init__(self, model, X, y, cv, pop_size=5, n_gen=10, n_jobs=1):

        self.model = model
        self.X = X
        self.y=y
        self.cv=cv
        self.toolbox = None
        self.creator = None
        self.stats = None

        # ga parameters
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_jobs=n_jobs

        

    def create_creator(self):
    # define fitness, individual CLASSES
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Individual", list, fitness=creator.FitnessMax)

        creator.create("FitnessMulti", base.Fitness, weights=(1,-1))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        return creator

    def eval_function(self, individual):
        ind = list(map(bool, individual))

        
        acccuracy = cross_val_score(estimator=self.model,
                                X=self.X[:,ind],
                                y=self.y,
                                cv=self.cv,
                                scoring='accuracy'
        ).mean().round(4)

        size = np.sum(ind)

        return (acccuracy, size)


    def _init_toolbox(self):
        toolbox = base.Toolbox()
        # Attribute generator
        random.seed(42)
        toolbox.register('attr_bool', random.randint, 0, 1)
        # Structure initializers
        toolbox.register('individual', tools.initRepeat, self.creator.Individual, toolbox.attr_bool, self.X.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox


    def register_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register('evaluate', self.eval_function)
        toolbox.register('mate', tools.cxPartialyMatched) #
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
        # toolbox.register('select', tools.selTournament, tournsize=5)
        toolbox.register('select', tools.selNSGA2)
        return toolbox
    
    def eaMuPlusLambda(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None,dynamic_pb=False, verbose=__debug__):
 
        # Logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'fitness', 'size']
        logbook.chapters['fitness'].header = "min", "avg", "max"
        logbook.chapters['size'].header = "min", "avg", "max"

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population

            if dynamic_pb:
                mutpb = gen/(ngen+1)
                cxpb = 1-mutpb

            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook


    def eaMuCommaLambda(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None,dynamic_pb=False, verbose=__debug__):
    
        assert lambda_ >= mu, "lambda must be greater or equal to mu."

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'fitness', 'size']
        logbook.chapters['fitness'].header = "min", "avg", "max"
        logbook.chapters['size'].header = "min", "avg", "max"

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population
            if dynamic_pb:
                mutpb = gen/(ngen+1)
                cxpb = 1-mutpb


            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
        return population, logbook


    def eaSimpleWithElitism(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, dynamic_pb=False):

        logbook = tools.Logbook()
        # Logbook
        logbook.header = ['gen', 'nevals', 'fitness', 'size']
        logbook.chapters['fitness'].header = "min", "avg", "max"
        logbook.chapters['size'].header = "min", "avg", "max"

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is None:
            raise ValueError("halloffame parameter must not be empty!")

        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - hof_size)

            if dynamic_pb:
                mutpb = gen/(ngen+1)
                cxpb = 1-mutpb
                

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # add the best back to population:
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook 

    def run(self):
        self.creator = self.create_creator()
        self.toolbox = self.register_toolbox()


        pool = Pool(self.n_jobs)
        self.toolbox.register('map', pool.map)


        pop = self.toolbox.population(n=self.pop_size)

        hof   = tools.HallOfFame(1)
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(lambda ind: sum(ind))
        self.stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        # Statistics
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)



        # Evolution
        pop, log = self.eaMuCommaLambda(pop, self.toolbox, mu=self.pop_size, lambda_=self.pop_size*3,
                                                cxpb=0.5,   mutpb=0.3, ngen=self.n_gen, 
                                                stats=self.stats, halloffame=hof, verbose=True, dynamic_pb=False)
        
        print("Best individual has fitness and size: %s" % ( hof[0].fitness))
        print()


        fig, axes = plt.subplots(1,2, figsize=(15,7))

        for i, (chp_name,chp) in enumerate(log.chapters.items()):
            legends = []
            for score in chp[0].keys():
                if (score != 'gen') and (score != 'nevals'): 
                    axes[i].plot(chp.select('gen'), chp.select(score))
                    axes[i].set(xlabel='gen', ylabel=chp_name)
                    legends.append(score)
                    
            axes[i].legend(legends)
        plt.show()

        return pop, log, hof[0]






if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.datasets import make_classification

    # Generating a df with 2 classes, 20 informative features out of 300 (10 of which redundant), 625 total samples 
    X,y = make_classification(n_classes=2, n_samples=625, n_features=200, n_informative=20, n_redundant=10, shuffle=True, random_state=42)

    # indipendent test validation separated
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, test_size=.2, random_state= 42)

    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')

    # defining model
    model = Pipeline([  ('scaler', MinMaxScaler()),
                            ('clf', LinearSVC(max_iter=5000))
                        ])

    # monte carlo cv for fitness evaluation
    oob = StratifiedShuffleSplit(n_splits=10, test_size=.3, random_state=42)

    # Setting GASelection class
    gas = GASelection(model=model, X=X_train, y=y_train, cv=oob, pop_size=5, n_gen=25, n_jobs=4)

    # Running GA
    pop, log, best = gas.run()

    best_bool = list(map(bool,best))
    print()

    print()
    print('### Test performances ###')

    model.fit(X_train, y_train)
    print(f'> ALL FEATURES Test partition accuracy: {model.score(X_test, y_test):.4f}')
    print()
    model.fit(X_train[:, best_bool], y_train)
    print(f'> GA SELECTED FEATURES partition accuracy: {model.score(X_test[:, best_bool], y_test):.4f}')



    