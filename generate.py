import numpy as np
import os
import time
import pickle
from copy import deepcopy
import sys
from catalystid import catalystID, argsObject                
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing        

args = argsObject()

args.checkpoint_path = "painn_h1024_bs4x8_is2re_all.pt"

args.config_yml_path = "/home/e/esargent/felixt/softwares/ocp/configs/is2re/all/painn/painn_h1024_bs8x4.yml"

args.bulk_db = "slabs_new.pkl"

args.adsorbate_db = "adsorbates_new.pkl"

args.output_dir = sys.argv[1]

args.adsorbate_index = 5 # CO

test = False

n_bulks = 100 #len(pickle.load(open(args.bulk_db, "rb")))

n_iter = 50
pop_size = 20
survive_size = 20
n_children = 20
n_newcomers = pop_size - survive_size - n_children  

gaussian = True
miller_max = 5

bulk_prob = 0.5
miller_prob = 0.5
term_prob = 0.5
site_prob = 0.5

miller_std = 1.5
min_prob = 1e-5

os.makedirs(args.output_dir, exist_ok = True)

f = open(args.output_dir + "/out_gen.txt","w")

# Display initialization

# plt.ion()
plt.figure()
plt.plot([],'r',label = 'Best value')
plt.plot([],'k',label = 'Avg value')
plt.plot([],'b',label = 'Worst value')
plt.xlabel('Epoch')
plt.ylabel('Adsorption energy')
plt.legend(loc=4)
plt.savefig(args.output_dir + "/progress_cur.png")

# Generate the population
pop = []
for i in range(pop_size):
    miller = [0,0,0]
    while all(np.array(miller) == 0) or any(abs(np.array(miller)) > miller_max):
        if gaussian:
            miller = [int(a) for a in np.rint(np.random.normal(0,miller_std,3))]
        else:
            miller = [int(a) for a in np.rint(np.random.randint(-miller_max, miller_max,3))]
    pop.append(catalystID(id=np.random.randint(0, n_bulks), miller=miller, term = np.random.randint(0, 10), site = np.random.randint(0, 10), args=args))

print("Initial pop:", file=f)
for cat in pop:
    print(cat, file=f)

pop = set(pop)

best = [0]
worst = [0]
avg = [0]
times = []
for i in range(n_iter):

    t = time.time()
    print("New Generation:", i, "---------------------", file=f, flush=True)

    # Evaluate
    p = []
    def evaluate(cat):
        num = multiprocessing.current_process()
        print("Thread started:", num, file=f, flush=True)
        cat.value
        print("Thread ended", num, cat, cat.timing, cat.folder, file=f, flush=True)
        return cat 

    pool = Pool()
    pop = pool.map(evaluate, pop)
    print("Pool DONE", file=f, flush=True)

    # Erase if too large
    idx = []
    for j, cat in enumerate(pop):
        if np.isnan(cat.value):
            pop[j].delete_folder()
        else:
            idx.append(j)

    pop = np.array(pop)[idx]

    p = np.array([cat.value for cat in pop])
    
    p = np.array(p)

    p_abs = p

    best.append(max(best[-1], p.max()))
    avg.append(np.mean(p))
    worst.append(p.min())
    
    # Display

    plt.plot(best,'r',label = 'Best Value')
    plt.plot(avg,'k',label = 'Avg Value')
    plt.plot(worst,'b',label = 'Worst Value')

    # Value determined
    v = (min_prob*np.sum(p) - p.min()) / (min_prob*len(p) - 1)

    if p.min() != p.max():
        p = p - v
        p = p**2 # Focusing on best results
        p = p / np.sum(p)
    else:
        p = np.ones(len(p))/len(p)

    # Select

    print("Chances of survival:", file=f, flush=True)
    idx = np.argsort(p)
    inv = np.argsort(idx)

    # # Postion determined
    # p = inv / np.sum(inv)
    
    for j in idx:
        print(pop[j], p[j], p_abs[j], pop[j].folder, file=f)
    print(file=f)

    idx = np.random.choice(range(len(pop)), survive_size, False, p)

    to_erase = list(set(range(len(pop))) - set(idx))

    # Erase excluded catalysts
    for j in to_erase:
        pop[j].delete_folder()

    pop = np.array(pop)[idx]
    
    print("Survivors:", file=f)
    for cat in pop:
        print(cat, file=f)
    print(file=f)
        
    # Reproduce

    couples = [np.random.choice(pop, 2, False, p[idx]/np.sum(p[idx])) for j in range((n_children))]
    
    babies = np.array([a.reproduce(b) for a,b in couples])

    print("Families:", file=f, flush=True)

    for j, b in enumerate(babies):
        print(couples[j][0],"+",couples[j][1],"=",b, file=f)
    print(file=f)
        
    # pop = np.concatenate((pop, babies)) 
    
    # Mutate

    print("Final pop", file=f)

    for cat in babies:
        old_cat = deepcopy(cat)
        cat.mutate(bulk_prob, miller_prob, term_prob, site_prob, miller_max = miller_max, n_bulks=n_bulks)
        print(old_cat, "-->", cat, file=f)
    print(file=f, flush=True)

    pop = np.concatenate((pop, babies))

    # Newcomers

    for j in range(n_newcomers):
        miller = [0,0,0]
        while all(np.array(miller) == 0) or any(abs(np.array(miller)) > miller_max):
            if gaussian:
                miller = [int(a) for a in np.rint(np.random.normal(0,2,3))]
            else:
                miller = [int(a) for a in np.rint(np.random.randint(-miller_max, miller_max,3))]

        pop = np.append(pop, catalystID(id=np.random.randint(0, n_bulks), miller=miller, term = np.random.randint(0, 10), site = np.random.randint(0, 10), args=args))

    times.append(time.time() - t)

#plt.ioff()
plt.savefig(args.output_dir + "/progress.png")
pickle.dump(best, open(args.output_dir + "/best.pkl","wb"))
pickle.dump(times, open(args.output_dir + "/times.pkl","wb"))
#plt.show()
