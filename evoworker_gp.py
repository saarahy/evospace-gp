#Archivos importados para el algoritmo
import operator
import csv
import funcEval
import numpy as np
import neatGPLS
import init_conf
import os.path
from deap import base
from deap import creator
from deap import tools
from deap import gp
import gp_conf as neat_gp
from my_operators import safe_div, mylog, mypower2, mypower3, mysqrt, myexp

#Imports de evospace
import random, time
import evospace
import xmlrpclib
import jsonrpclib
import cherrypy_server


pset = gp.PrimitiveSet("MAIN", 13)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(np.cos, 1)
pset.addPrimitive(np.sin, 1)
#pset.addPrimitive(myexp, 1)
pset.addPrimitive(mylog, 1)
pset.addPrimitive(mypower2, 1)
pset.addPrimitive(mypower3, 1)
pset.addPrimitive(mysqrt, 1)
pset.addPrimitive(np.tan, 1)
pset.addPrimitive(np.tanh, 1)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
pset.renameArguments(ARG0='x0',ARG1='x1', ARG2='x2', ARG3='x3', ARG4='x4', ARG5='x5', ARG6='x6', ARG7='x7',  ARG8='x8', ARG9='x9',  ARG10='x10',  ARG11='x11',  ARG12='x12')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessTest", base.Fitness, weights=(-1.0,))
creator.create("Individual", neat_gp.PrimitiveTree, fitness=creator.FitnessMin, fitness_test=creator.FitnessTest)

def getToolBox(config):
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=6)
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", init_conf.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # Operator registering
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", neat_gp.cxSubtree)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=6)
    toolbox.register("mutate", neat_gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    #toolbox.register("evaluate", evalSymbReg, points=data_[0])
    #toolbox.register("evaluate_test", evalSymbReg, points=data_[1])

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox


def initialize(config):
    pop = getToolBox(config).population(n=config["POPULATION_SIZE"])
    server = evospace.Population("pop")#jsonrpclib.Server(config["SERVER"])
    server.initialize()
    #server.initialize(None)

    sample = [{"chromosome":str(ind), "id":None, "fitness":{"DefaultContext":0.0}} for ind in pop]
    init_pop = {'sample_id': 'None' , 'sample':   sample}
    server.put_sample(init_pop)
    #server.putSample(init_pop)


def evalSymbReg(individual, points, toolbox):
    func = toolbox.compile(expr=individual)
    vector = points[13]
    data_x=np.asarray(points)[:13]
    vector_x=func(*data_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(vector_x, np.ndarray):
            for e in range(len(vector_x)):
                if np.isnan(vector_x[e]) or np.isinf(vector_x[e]):
                    vector_x[e] = 0.
    result = np.sum((vector_x - vector)**2)
    return np.sqrt(result/len(points[0])),

def data_(n_corr,p, toolbox):
    n_archivot='./data_corridas/Housing/test_%d_%d.txt'%(p,n_corr)
    n_archivo='./data_corridas/Housing/train_%d_%d.txt'%(p,n_corr)
    if not (os.path.exists(n_archivo) or os.path.exists(n_archivot)):
        direccion="./data_corridas/Housing/housing.txt"
        with open(direccion) as spambase:
            spamReader = csv.reader(spambase,  delimiter=' ', skipinitialspace=True)
            num_c = sum(1 for line in open(direccion))
            num_r = len(next(csv.reader(open(direccion), delimiter=' ', skipinitialspace=True)))
            Matrix = np.empty((num_r, num_c,))
            for row, c in zip(spamReader, range(num_c)):
                for r in range(num_r):
                    try:
                        Matrix[r, c] = row[r]
                    except ValueError:
                        print 'Line {r} is corrupt', r
                        break
        if not os.path.exists(n_archivo):
            long_train=int(len(Matrix.T)*.7)
            data_train1 = random.sample(Matrix.T, long_train)
            np.savetxt(n_archivo, data_train1, delimiter=",", fmt="%s")
        if not os.path.exists(n_archivot):
            long_test=int(len(Matrix.T)*.3)
            data_test1 = random.sample(Matrix.T, long_test)
            np.savetxt(n_archivot, data_test1, delimiter=",", fmt="%s")
    with open(n_archivo) as spambase:
        spamReader = csv.reader(spambase,  delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivo))
        num_r = len(next(csv.reader(open(n_archivo), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, range(num_c)):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print 'Line {r} is corrupt' , r
                    break
        data_train=Matrix[:]
    with open(n_archivot) as spambase:
        spamReader = csv.reader(spambase,  delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivot))
        num_r = len(next(csv.reader(open(n_archivot), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, range(num_c)):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print 'Line {r} is corrupt' , r
                    break
        data_test=Matrix[:]
    #return data_train,data_test
    toolbox.register("evaluate", evalSymbReg, points=data_train, toolbox=toolbox)
    toolbox.register("evaluate_test", evalSymbReg, points=data_test, toolbox=toolbox)

def evolve(sample_num, config):
    #random.seed(64)
    toolbox = getToolBox(config)
    start = time.time()
    problem=config["PROBLEM"]
    direccion=config["DIRECCION"]
    n_corr=config["n_corr"]
    n_prob=config["n_problem"]



    server = evospace.Population("pop")
    #server = jsonrpclib.Server(config["SERVER"])

    evospace_sample = server.get_sample(config["SAMPLE_SIZE"])
    #evospace_sample = server.getSample(config["SAMPLE_SIZE"])

    pop = [ creator.Individual(neat_gp.PrimitiveTree.from_string(cs['chromosome'], pset)) for cs in evospace_sample['sample']]

    cxpb = config["CXPB"]#0.7  # 0.9
    mutpb = config["MUTPB"]#0.3  # 0.1
    ngen = config["WORKER_GENERATIONS"]#50000
    params = config["PARAMS"]
    neat_cx = config["neat_cx"]
    neat_alg = config["neat_alg"]
    neat_pelit = config["neat_pelit"]
    neat_h = config["neat_h"]
    funcEval.LS_flag = config["LS_FLAG"]
    LS_select = config["LS_SELECT"]
    funcEval.cont_evalp = 0
    num_salto = config["num_salto"]
    cont_evalf = config["cont_evalf"]
    SaveMatrix = config["save_matrix"]
    GenMatrix = config["gen_matrix"]

    data_(n_corr, n_prob, toolbox)

    begin =   time.time()
    print "inicio del proceso"
    pop, log = neatGPLS.neat_GP_LS(pop, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h, neat_pelit,
                                   funcEval.LS_flag, LS_select, cont_evalf, num_salto, SaveMatrix, GenMatrix, pset,
                                   n_corr, n_prob, params, direccion, problem, stats=None, halloffame=None, verbose=True)
    # Evaluate the entire population
    #fitnesses = map(toolbox.evaluate, pop)
    # for ind, fit in zip(pop, fitnesses):
    #     ind.fitness.values = fit
    #
    #
    # total_evals = len(pop)
    # best_first   = None
    # # Begin the evolution
    #
    # for g in range(config["WORKER_GENERATIONS"]):
    #     # Select the next generation individuals
    #     offspring = toolbox.select(pop, len(pop))
    #     # Clone the selected individuals
    #     offspring = map(toolbox.clone, offspring)
    #
    #     # Apply crossover and mutation on the offspring
    #     for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #         if random.random() < config["CXPB"]:
    #             toolbox.mate(child1, child2)
    #             del child1.fitness.values
    #             del child2.fitness.values
    #
    #     for mutant in offspring:
    #         if random.random() < config["MUTPB"]:
    #             toolbox.mutate(mutant)
    #             del mutant.fitness.values
    #
    #     # Evaluate the individuals with an invalid fitness
    #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    #     fitnesses = map(toolbox.evaluate, invalid_ind)
    #     for ind, fit in zip(invalid_ind, fitnesses):
    #         ind.fitness.values = fit
    #
    #     total_evals+=len(invalid_ind)
    #     #print "  Evaluated %i individuals" % len(invalid_ind),
    #
    #     # The population is entirely replaced by the offspring
    #     pop[:] = offspring
    #
    #     # Gather all the fitnesses in one list and print the stats
    #     fits = [ind.fitness.values[0] for ind in pop]
    #
    #     #length = len(pop)
    #     #mean = sum(fits) / length
    #     #sum2 = sum(x*x for x in fits)
    #     #std = abs(sum2 / length - mean**2)**0.5
    #
        # best = max(fits)
        # if not best_first:
        #     best_first = best
        #
        # if best >= config["CHROMOSOME_LENGTH"]:
        #     break
    #
    #     #print  "  Min %s" % min(fits) + "  Max %s" % max(fits)+ "  Avg %s" % mean + "  Std %s" % std
    #
    # print "-- End of (successful) evolution --"
    #
    putback =  time.time()
    #
    sample = [ {"chromosome":str(ind),"id":None, "fitness":{"DefaultContext":ind.fitness.values[0]} } for ind in pop]
    #print sample
    evospace_sample['sample'] = sample
    server.put_sample(evospace_sample)
    #server.putSample(evospace_sample)
    best_ind = tools.selBest(pop, 1)[0]
    #
    best = config["CHROMOSOME_LENGTH"], [config["CHROMOSOME_LENGTH"], sample_num, round(time.time() - start, 2),
                                         round(begin - start, 2), round(putback - begin, 2),
                                         round(time.time() - putback, 2), best_ind]
    return best
    #

def work(params):
    worker_id = params[0][0]
    config = params[0][1]
    #server = jsonrpclib.Server(config["SERVER"])
    results = []
    for sample_num in range(config["MAX_SAMPLES"]):
        # if int(server.found(None)):
        #     break
        # else:
            gen_data = evolve(sample_num, config)
            # if gen_data[0]:
            #     evospace.found_it(None)
            results.append([worker_id] + gen_data[1])
    return results

