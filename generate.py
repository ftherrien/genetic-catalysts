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
import argparse

def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c",dest="checkpoint_path",type=str, default="painn_h1024_bs4x8_is2re_all.pt", help="Pretrained model weights")
    parser.add_argument("-y",dest="config_yml_path",type=str, default="/home/e/esargent/felixt/softwares/ocp/configs/is2re/all/painn/painn_h1024_bs8x4.yml", help="Config file")
    parser.add_argument("-b",dest="bulk_db",type=str, default="slabs_new.pkl", help="Pickle containing materials")
    parser.add_argument("-a",dest="adsorbate_db",type=str, default="adsorbates_new.pkl", help="Pickle containing adsorbates")
    parser.add_argument("-o",dest="output_dir",type=str, default="out", help="Output directory")
    parser.add_argument("-n",dest="adsorbate_index",type=int, default=5, help="Adsorbate number")

    parser.add_argument("-G",dest="gen",type=int, default=None, help="Generation number")
    parser.add_argument("-I",dest="ID",type=int, default=None, help="ID number")

    args = parser.parse_args()

    return args


args = read_options()

# args = argsObject()

# args.checkpoint_path = "painn_h1024_bs4x8_is2re_all.pt"

# args.config_yml_path = "/home/e/esargent/felixt/softwares/ocp/configs/is2re/all/painn/painn_h1024_bs8x4.yml"

# args.bulk_db = "slabs_new.pkl"

# args.adsorbate_db = "adsorbates_new.pkl"

# args.output_dir = sys.argv[1]

# args.adsorbate_index = 5 # CO

test = False

n_bulks = 100 #len(pickle.load(open(args.bulk_db, "rb")))

n_iter = 50
pop_size = 20
survive_size = 10
n_children = 10
n_newcomers = pop_size - survive_size - n_children  

gaussian = True
miller_max = 5

bulk_prob = 0.5
miller_prob = 0.5
term_prob = 0.5
site_prob = 0.5

miller_std = 1.5
min_prob = 1e-5


if args.gen is None and args.ID is None:

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
    while pop < pop_size:
        miller = [0,0,0]
        while all(np.array(miller) == 0) or any(abs(np.array(miller)) > miller_max):
            if gaussian:
                miller = [int(a) for a in np.rint(np.random.normal(0,miller_std,3))]
            else:
                miller = [int(a) for a in np.rint(np.random.randint(-miller_max, miller_max,3))]
        pop.append(catalystID(id=np.random.randint(0, n_bulks), miller=miller, term = np.random.randint(0, 10), site = np.random.randint(0, 10), args=args))
        pop = list(set(pop))

    os.makedirs(args.output_dir+"/0", exist_ok = True)

    print("Initial pop:", file=f)
    for i, cat in enumerate(pop):
        print(cat, file=f)
        os.makedirs(args.output_dir+"/0/%d"%(i), exist_ok = True)
        pickle.dump(cat, open(args.output_dir+"/0/%d/cat.pickle"%(i),"wb"))

    best = [0]
    worst = [0]
    avg = [0]
    times = []

    pickle.dump(cat, open(args.output_dir+"progress_info.pickle","wb"))

else:

    # Read local cat
    cat = pickle.load(open(args.output_dir+"/%d/%d/cat.pickle"%(args.gen, args.ID),"rb"))

    cat.value

    pickle.dump(cat, open(args.output_dir+"/%d/%d/cat_value.pickle"%(args.gen, args.ID),"wb"))

    # Check if others are done

    pickle_list = list(iglob(args.output_dir+"/%d/*/cat_value.pickle"%(args.gen)))

    if len(pickle_list) == pop_size:

        pop = [pickle.load(open(cat,"rb")) for cat in pickle_list]

        f = open(args.output_dir + "/out_gen.txt","w")
        
        t = time.time()
        
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
        
        # # Erase excluded catalysts
        # for j in to_erase:
        #     pop[j].delete_folder()
        
        pop = np.array(pop)[idx]
        
        print("Survivors:", file=f)
        for j, cat in enumerate(pop):
            os.makedirs(args.output_dir+"/%d/%d"%(args.gen+1,j+pop_size-survive_size), exist_ok = True)
            pickle.dump(cat, open(args.output_dir+"/%d/%d/cat_value.pickle"%(args.gen+1,j+pop_size-survive_size),"wb"))
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
        
        unique_new = set(babies) - set(babies).intersection(set(pop))

       # Newcomers
        
        while unique_new < pop_size - survive_size:
            miller = [0,0,0]
            while all(np.array(miller) == 0) or any(abs(np.array(miller)) > miller_max):
                if gaussian:
                    miller = [int(a) for a in np.rint(np.random.normal(0,2,3))]
                else:
                    miller = [int(a) for a in np.rint(np.random.randint(-miller_max, miller_max,3))]
        
            unique_new.add(catalystID(id=np.random.randint(0, n_bulks), miller=miller, term = np.random.randint(0, 10), site = np.random.randint(0, 10), args=args))

            unique_new = set(unique_new) - set(unique_new).intersection(set(pop))
        

        for j, cat in enumerate(unique_new):
            os.makedirs(args.output_dir+"/%d/%d"%(args.gen+1,j), exist_ok = True)
            pickle.dump(cat, open(args.output_dir+"/%d/%d/cat.pickle"%(args.gen+1,j),"wb"))

        times.append(time.time() - t)
        
        print("New Generation:", i, "---------------------", file=f, flush=True)

#plt.ioff()
plt.savefig(args.output_dir + "/progress.png")
pickle.dump(best, open(args.output_dir + "/best.pkl","wb"))
pickle.dump(times, open(args.output_dir + "/times.pkl","wb"))
#plt.show()
