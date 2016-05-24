import evoworker_gp as evoworker
import cloud, time, yaml


config = yaml.load(open("conf/conf.yaml"))

start = time.time()

# init_job = cloud.call(onemax.initialize, config=config,  _type='s1', _env="deap")

params = [(i, config) for i in range(2)]

# jids = cloud.map(onemax.work, params, _type='s1',_depends_on= init_job )
# results_list = cloud.result(jids)
#
# print time.time()-start
#
# for r in results_list:
#     for a in r:
#         print a

evoworker.initialize(config)
result=evoworker.work(params)
if result:
    print 'finished'

