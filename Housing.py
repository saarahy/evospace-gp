import operator
import random
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

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=0, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", init_conf.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
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

def energy_coolng(n_corr,p):
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
    toolbox.register("evaluate", evalSymbReg, points=data_train)
    toolbox.register("evaluate_test", evalSymbReg, points=data_test)


def main(n_corr, p):
    problem = "Housing"
    direccion="./data_corridas/Housing/train_%d_%d.txt"
    energy_coolng(n_corr,p)

    pop_size = 100
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", neat_gp.cxSubtree)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", neat_gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(3)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    cxpb = 0.7#0.9
    mutpb = 0.3#0.1
    ngen = 50000
    params = ['best_of_each_specie', 2, 'yes']
    neat_cx = False
    neat_alg = True
    neat_pelit = 0.5
    neat_h = 0.15
    funcEval.LS_flag = True
    LS_select = 9
    funcEval.cont_evalp = 0
    num_salto = 500
    cont_evalf = 100000
    SaveMatrix = True
    GenMatrix = False

    pop, log = neatGPLS.neat_GP_LS(pop, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h, neat_pelit, funcEval.LS_flag, LS_select, cont_evalf, num_salto, SaveMatrix, GenMatrix, pset,n_corr, p, params, direccion,problem,stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof


def run(number, problem):
    n = 1
    while n <= number:
        main(n, problem)
        n += 1
#EC
#40f
#9530 LSHS-Sp cx-subarbol neat-gp 40f probabilidades como el std e inicializacion
#9531 LSHS-Sp cx-neat neat-gp probabilidades como el std e inicializacion


#Housing
#40f
#5000 Gp Standard trono
#5101 LSHS 40f
#5200 neat-alg cx-subarbol probabilidades como el std e inicializacion
#5201 neat-alg cx subarbol probabilidades como neat-GP
#5210 neat-alg cx-neat probababilidades como neat-GP
#5700 best set cx subarbol 40f
#5800 random 40f
#5900 best random 40f
#52101 listo
#5220 listo
#5230 listo
if __name__ == "__main__":
    n = 1
    while n < 11:
        main(n, 5901)
        n += 1
