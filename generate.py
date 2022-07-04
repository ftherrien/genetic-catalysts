import numpy as np
from ocdata.structure_sampler import StructureSampler
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import ase.io
from ase.optimize import BFGS
from ase.build import fcc100, add_adsorbate, molecule
import os
from ase.constraints import FixAtoms
import time
import pickle
from copy import deepcopy
from multiprocessing import Pool
import pickle

from contextlib import contextmanager
import sys

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

@contextmanager
def suppress_out():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class argsObject():

    def __init__(self):
        self.seed = None#Random seed for sampling
        
        # input and output
        self.bulk_db = "bulk_db_flat.pkl" # Underlying db for bulks
        self.adsorbate_db = "adsorbate_db.pkl" # Underlying db for adsorbates
        self.output_dir = "outputs"  # Root directory for outputs
        
        # for optimized (automatically try to use optimized if this is provided)
        self.precomputed_structures = None # Root directory of precomputed structures
        
        # args for enumerating all combinations:
        self.enumerate_all_structures = True # Find all possible structures given a specific adsorbate and a list of bulks
        self.adsorbate_index = None # Adsorbate index (int)
        
        self.verbose = False # Log detailed info


class catalystID():

    def __init__(self, id=0, miller=[1,0,0], term = 0, site = 0):

        self.bulk_index = id

        if all(np.array(miller) == 0):
            raise ValueError("Miller indices of [0,0,0]")
            
        self.miller = miller

        self.term = term

        self.site = site

        self._value = None

    def reproduce(self, partner):

        baby = catalystID()

        baby.bulk_index = np.random.choice([self.bulk_index, partner.bulk_index])
        
        baby.miller = deepcopy(np.random.choice([self, partner]).miller)

        baby.term = np.random.choice([self.term, partner.term])

        baby.site = np.random.choice([self.site, partner.site])

        return baby

    def mutate(self, bulk_prob, miller_prob, term_prob, site_prob):

        if np.random.random() < bulk_prob:
            self.bulk_index = np.random.randint(0, n_bulks)
            self._value = None

        if np.random.random() < miller_prob:
            enter = True
            while all(np.array(self.miller) == 0) or enter:
                enter = False
                self.miller += np.array([int(a) for a in np.rint(np.random.normal(0,1,3))]) #np.random.randint(-1,2,3)
            self._value = None
            
        if np.random.random() < term_prob:
            self.term += 1
            self._value = None

        if np.random.random() < site_prob:
            self.site += 1
            self._value = None

    @property
    def value(self):

        if self._value is None:

            if not test:
                t = time.time()
                #with suppress_out():
                job = StructureSampler(args, self)
                struc = job.run()
                
                print("Generation time", time.time() - t)
                
                with suppress_out():
                    calc = OCPCalculator(config_yml=config_yml_path, checkpoint=checkpoint_path)
                
                struc.calc = calc
                
                self._value = -struc.get_potential_energy()
                
                print("Output value:", self._value)

            else:
                
                self._value = 1/(0.01 + np.linalg.norm(self.miller - np.array([1,1,3]))) + 1/(0.01 + abs(self.bulk_index - 17))
            
        return self._value

    def __str__(self):

        return "(%s, %s, %s, %s)"%(str(self.bulk_index), str(self.miller), str(self.term), str(self.site))

    def __rep__(self):

        return self.__str__()
    
checkpoint_path = "pretrained/painn_h1024_bs4x8_is2re_all.pt"

config_yml_path = "../ocp/configs/is2re/all/painn/painn_h1024_bs8x4.yml"            
        
args = argsObject()

args.bulk_db = "../slabs_new.pkl"

args.adsorbate_db = "../adsorbates_new.pkl"

args.output_dir = "./outest"

args.adsorbate_index = 5 # CO

test = False

n_bulks = 100 #len(pickle.load(open(args.bulk_db, "rb")))

n_iter = 10
pop_size = 20
survive_size = 10

bulk_prob = 0.5
miller_prob = 0.5
term_prob = 0.5
site_prob = 0.5

miller_std = 1.5
min_prob = 1e-5

f = open("out_gen.txt","w")

# Display initialization

plt.ion()
plt.figure()
plt.plot([],'r',label = 'Best value')
plt.plot([],'k',label = 'Avg value')
plt.plot([],'b',label = 'Worst value')
plt.xlabel('Epoch')
plt.ylabel('Adsorption energy')
plt.legend(loc=4)

# Generate the population
pop = []
for i in range(pop_size):
    miller = [0,0,0]
    while all(np.array(miller) == 0):
        miller = [int(a) for a in np.rint(np.random.normal(0,miller_std,3))]
    pop.append(catalystID(id=np.random.randint(0, n_bulks), miller=miller, term = np.random.randint(0, 10), site = np.random.randint(0, 10)))

print("Initial pop:", file=f)
for cat in pop:
    print(cat, file=f)

best = []
worst = []
avg = []
for i in range(n_iter):

    print("New Generation:", i, "---------------------", file=f)

    # Evaluate
    p = []
    def evaluate(cat):
        return cat.value

    pool = Pool()
    p = pool.map(evaluate, pop)
    
    # p = np.array([cat.value for cat in pop])
    
    p = np.array(p)

    p_abs = p

    best.append(p.max())
    avg.append(np.mean(p))
    worst.append(p.min())
    
    # Display

    plt.plot(best,'r',label = 'Best Value')
    plt.plot(avg,'k',label = 'Avg Value')
    plt.plot(worst,'b',label = 'Worst Value')
    plt.pause(0.1)

    # Value determined
    v = (min_prob*np.sum(p) - p.min()) / (min_prob*len(p) - 1)

    if p.min() != p.max():
        p = p - v
        p = p**2 # Focusing on best results
        p = p / np.sum(p)
    else:
        p = np.ones(len(p))/len(p)

    # Select

    print("Chances of survival:", file=f)
    idx = np.argsort(p)
    inv = np.argsort(idx)

    # # Postion determined
    # p = inv / np.sum(inv)
    
    for j in idx:
        print(pop[j], p[j], p_abs[j], file=f)
    print(file=f)

    idx = np.random.choice(range(pop_size), survive_size, False, p)

    pop = np.array(pop)[idx]
    
    print("Survivors:", file=f)
    for cat in pop:
        print(cat, file=f)
    print(file=f)
        
    # Reproduce

    couples = [np.random.choice(pop, 2, False, p[idx]/np.sum(p[idx])) for j in range((pop_size-survive_size))]
    
    babies = np.array([a.reproduce(b) for a,b in couples])

    print("Families:", file=f)

    for j, b in enumerate(babies):
        print(couples[j][0],"+",couples[j][1],"=",b, file=f)
    print(file=f)
        
    # pop = np.concatenate((pop, babies)) 
    
    # Mutate

    print("Final pop", file=f)

    for cat in babies:
        old_cat = deepcopy(cat)
        cat.mutate(bulk_prob, miller_prob, term_prob, site_prob)
        print(old_cat, "-->", cat, file=f)
    print(file=f, flush=True)

    pop = np.concatenate((pop, babies))
    
plt.ioff()
plt.savefig("progress.png")
plt.show()
