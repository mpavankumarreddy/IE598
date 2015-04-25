import random                   # @config
import numpy                    # @config
import networkx as nx           # @config
import itertools                # @config
from scipy.stats import beta    # @config
import matplotlib.pyplot as plt     # @config


n = 2000    # @config
param_alpha = 6   # @config
param_beta = 2    # @config
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
    return 0.1 + 0.9*numpy.random.beta(param_alpha, param_beta, (m,1))


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
            G.add_edge(worker, task, msg_t_w = None, msg_w_t = None, log_msg_t_w = None, log_msg_w_t = None, Aij = aij, t_id = nl[j], w_id = mr[j])

print 'Generating random graph here'
configuration_model_connect(G, n, m, l, r)
print 'Done generating graph'

# A simple quantization mechanism to get a discrete probability distribution over the beta distribution

domain_size_p = 8 # number of points in the discrete distribution
domain_p = numpy.linspace(0, 1, domain_size_p + 2)[1:-1]

dist_p = numpy.array([beta.pdf(i, param_alpha, param_beta) for i in domain_p])
dist_p = dist_p / sum(dist_p)

domain_t = [-1, 1]
domain_size_t = 2

# a little bit optimisations
t_eq = numpy.multiply(domain_p, dist_p)
t_ne = numpy.multiply(1 - domain_p, dist_p)

# int(True) = 1, int(False) = 0
t_vals = [t_ne, t_eq]


# should we make messages log? Lets see, how big they are after first iteration

for e in G.edges_iter(data=True):
    # update msg_t_w to be lookup table of 2 value and initialise these messages to 1
    # WARNING: here [1, 1] corresponds to -1 and 1 values of ti. Didnot make it a dict for performance reasons
    e[2]['msg_t_w'] = numpy.array([1.0, 1.0])
    # update msg_w_t to be lookup table of domain_size_p and no initialisation required here
    # sorted keys order
    e[2]['msg_w_t'] = numpy.array([1.0]*domain_size_p)


def worker_to_task():
    # one bp loop - later incorporate into a while loop
    print 'worker to task'
    for e in G.edges_iter(data=True):
        # worker to task message updates

        worker_j = workers[e[2]['w_id']]

        adj_task_nodes = [out_edge['t_id'] for out_edge in G[worker_j].values()]
        adj_task_nodes.remove(e[2]['t_id'])

        vkjs = [G[tasks[i]][worker_j]['msg_t_w'] for i in adj_task_nodes]
        akjs = [G[tasks[i]][worker_j]['Aij'] for i in adj_task_nodes]

        for i in range(domain_size_p):
            p = domain_p[i]
            psi_k_j = [numpy.array([1-p, p])[::akj] for akj in akjs]
            e[2]['msg_w_t'][i] = (numpy.prod([numpy.dot(vkjs[k], psi_k_j[k]) for k in range(len(adj_task_nodes))]))


def task_to_worker():
    print 'task to worker'
    for e in G.edges_iter(data=True):
        # task to worker message updates

        task_i = tasks[e[2]['t_id']]

        adj_worker_nodes = [out_edge['w_id'] for out_edge in G[task_i].values()]
        adj_worker_nodes.remove(e[2]['w_id'])

        vkis = [G[workers[k]][task_i]['msg_w_t'] for k in adj_worker_nodes]
        aiks = [G[workers[k]][task_i]['Aij'] for k in adj_worker_nodes]


        for i in range(domain_size_t):
            ti = domain_t[i]

            # If they are equal, 1 else -1
            e[2]['msg_t_w'][i] = numpy.prod([numpy.dot(t_vals[(int(ti == aiks[l]))], vkis[l]) for l in range(len(adj_worker_nodes))])


def estimate_ti():
    t_hats = []
    for i in range(len(tasks)):
        t_hat_i = []

        task_i = tasks[i]

        adj_worker_nodes = [out_edge['w_id'] for out_edge in G[task_i].values()]

        vkis = [G[workers[k]][task_i]['msg_w_t'] for k in adj_worker_nodes]
        aiks = [G[workers[k]][task_i]['Aij'] for k in adj_worker_nodes]

        for i in range(domain_size_t):
            ti = domain_t[i]

            # If they are equal, 1 else -1
            t_hat_i.append(numpy.prod([numpy.dot(t_vals[(int(ti == aiks[l]))], vkis[l]) for l in range(len(adj_worker_nodes))]))

        t_hats.append(t_hat_i)

    return t_hats

def estimate_pj():
    w_hats = []
    for j in range(len(workers)):
        w_hat_j = []

        worker_j = workers[j]

        adj_task_nodes = [out_edge['t_id'] for out_edge in G[worker_j].values()]

        vkjs = [G[tasks[i]][worker_j]['msg_t_w'] for i in adj_task_nodes]
        akjs = [G[tasks[i]][worker_j]['Aij'] for i in adj_task_nodes]

        for i in range(domain_size_p):
            p = domain_p[i]
            psi_k_j = [numpy.array([1-p, p])[::akj] for akj in akjs]
            w_hat_j.append(numpy.prod([numpy.dot(vkjs[k], psi_k_j[k]) for k in range(len(adj_task_nodes))]))


        w_hats.append(w_hat_j)

    return w_hats


def evaluate_ti_estimates():
    # estimating task labels
    t_hats = estimate_ti()
    t_est = numpy.array([t_h.index(max(t_h))*2 - 1 for t_h in t_hats])

    t_act = numpy.array([task.label for task in tasks])
    print "num correct - " + str(sum(t_est == t_act)) + " out of " + str(len(t_est))

def evaluate_wj_estimates():
    # estimating worker reliability
    w_hats = estimate_pj()
    w_est = numpy.array([domain_p[w_h.index(max(w_h))] for w_h in w_hats])

    w_act = numpy.array([worker.reliability for worker in workers])

