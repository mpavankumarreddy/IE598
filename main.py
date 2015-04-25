import random                   # @config
import numpy                    # @config
import networkx as nx           # @config
import itertools                # @config


n = 2000    # @config
alpha = 6   # @config
beta = 2    # @config
max_l = 15  # @config

# n tasks with binary true labels i.e t = [t_1,...,t_n]
# t_i = {+1 w.p. 0.75, -1 else}
p_t = 0.75 # @config # probability that t_i is +1

# change seed to get different results
random.seed(1)

def gen_t_i():
    '''generates a random label with configured bernoulli distirbution'''
    if random.random() < p_t:
        return 1
    else:
        return -1

def gen_betarnd(m):
    '''
    returns a numpy array of m beta distributed random vars with
    configured alpha and beta
    '''
    return 0.1 + 0.9*numpy.random.beta(alpha, beta, (m,1))


class Task(object):
    '''A task in a crowdsourcing problem. Helper methods include -
    --------
    '''
    def __init__(self, task_id, label):
        self.task_id = task_id
        self.label = label


class Worker(object):
    '''Model for a worker in a crowdsourcing problem. Helper methods include -
    --------
    '''
    def __init__(self, worker_id, reliability):
        self.reliability = reliability
        self.worker_id = worker_id


tasks = [Task(i, gen_t_i()) for i in range(n)]

# A sample belief propagation on a random l-r
l = 8       # l - r random graph
r = 16      # l - r random graph

m = int((n*l) / r)    # a free parameter

worker_reliabilities = gen_betarnd(m)   # Note that this is a m x 1 matrix
#workers = [Worker(reliability) for reliability in worker_reliabilities]
workers = [Worker(i, worker_reliabilities[i][0]) for i in range(m)]


G = nx.Graph()
G.add_nodes_from(tasks)
G.add_nodes_from(workers)


'''
# works but shag. Can get stuck when very few edges remain to be connected
# ignore for now. Fix later

# configuration model to generate a random graph - basically not quite random
# adding random edges between tasks and workers
total_edges_max = l * n
total_edges = 0

while total_edges < total_edges_max:
    task_num = numpy.random.randint(0, n)       # uniform sampling in [low, high)
    worker_num = numpy.random.randint(0, m)

    while G.degree(tasks[task_num]) >= l:
        task_num = numpy.random.randint(0, n)
    while G.degree(workers[worker_num]) >= r:
        worker_num = numpy.random.randint(0, m)

    task = tasks[task_num]
    worker = workers[worker_num]
    if worker not in G[task]:
        G.add_edge(task, worker)
        total_edges += 1
        print total_edges
'''

def random_arr_permute(arr):
    '''
    Generates a random permutation of elements in the array arr using <whatever> algorithm. Swap x <-> arr[random(k-1)] assuming you are adding x at position k.
    '''
    for i in range(len(arr)):
        random_index = numpy.random.randint(0, i+1)     # to include i too
        arr[random_index], arr[i] = arr[i], arr[random_index]

def permuted_range_with_repetetitions(n, l):
    '''
    A custom method to give out a random permutation of numbers from 0 to n-1 each repeated l times'''
    nl = [i for i in range(n) for j in range(l)]
    random_arr_permute(nl)
    return nl

def configuration_model_connect(G, n, m, l, r):
    nl = permuted_range_with_repetetitions(n, l)
    mr = permuted_range_with_repetetitions(m, r)
    if n * l != m * r:
        print "Error"
        return
    for j in range(n*l):
        worker = workers[mr[j]]
        task = tasks[nl[j]]
        if worker not in G[task]:
            # generate responses based on Dawid-Skene model
            # worker.reliability seems to be an array yo!
            if random.random() < worker.reliability:
                aij = task.label
            else:
                aij = -task.label
            G.add_edge(worker, task, msg_t_w = None, msg_w_t = None, Aij = aij)

configuration_model_connect(G, n, m, l, r)
