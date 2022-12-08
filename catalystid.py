import numpy as np
from ocdata.structure_sampler import StructureSampler
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import os
import time
import pickle
from copy import deepcopy
import sys
from contextlib import contextmanager
import shutil

test = False

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

        self.config_yml_path = "." 

        self.checkpoint_path = "."


class catalystID():

    def __init__(self, id=0, miller=[1,0,0], term = 0, site = 0, args=None):

        self.bulk_index = id

        if all(np.array(miller) == 0):
            raise ValueError("Miller indices of [0,0,0]")
            
        self.miller = miller

        self.term = term

        self.site = site

        self._value = None

        self.timing = None

        self.folder = None

        self.args = args

    def reproduce(self, partner):

        baby = catalystID()

        baby.bulk_index = np.random.choice([self.bulk_index, partner.bulk_index])
        
        baby.miller = deepcopy(np.random.choice([self, partner]).miller)

        baby.term = np.random.choice([self.term, partner.term])

        baby.site = np.random.choice([self.site, partner.site])

        baby.args = self.args

        return baby

    def mutate(self, bulk_prob, miller_prob, term_prob, site_prob, miller_max=10, n_bulks=None):

        if n_bulks is None:
            n_bulks = len(pickle.load(open(self.args.bulk_db, "rb")))

        if np.random.random() < bulk_prob:
            self.bulk_index = np.random.randint(0, n_bulks)
            self._value = None

        if np.random.random() < miller_prob:
            new_miller = [0,0,0]
            while all(np.array(new_miller) == 0) or any(abs(np.array(new_miller)) > miller_max):
                new_miller = self.miller + np.array([int(a) for a in np.rint(np.random.normal(0,1,3))]) #np.random.randint(-1,2,3)
            self.miller = new_miller
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
                try:
                    job = StructureSampler(self.args, self)
                    struc, timing, location = job.run()
                except FileExistsError:
                    pass

                self.folder = location

                self.timing = timing

                print("Generation time", time.time() - t)

                print("Size:", len(struc))

                if len(struc) < 150:
                    with suppress_out():
                        calc = OCPCalculator(config_yml=self.args.config_yml_path, checkpoint=self.args.checkpoint_path)
                    
                    struc.calc = calc
                    
                    self._value = -struc.get_potential_energy()
                else:
                    print("Too large:", location)
                    self._value = np.nan

                print("Output value:", self._value)

            else:
                
                self._value = 1/(0.01 + np.linalg.norm(self.miller - np.array([1,1,3]))) + 1/(0.01 + abs(self.bulk_index - 17))
            
        return self._value

    def delete_folder(self):
        try:
            shutil.rmtree(self.folder)
        except FileNotFoundError:
            pass

    def __str__(self):

        return "(%s, %s, %s, %s)"%(str(self.bulk_index), str(self.miller), str(self.term), str(self.site))

    def __rep__(self):

        return self.__str__()

    def __eq__(self, other):

        return all([s == o for s,o in zip(self.miller, other.miller)]) and (self.term == other.term) and (self.site == other.site) and (self.bulk_index == other.bulk_index)

    def __hash__(self):

        return hash(tuple(self.miller)) ^ hash(self.term) ^ hash(self.site) ^ hash(self.bulk_index)
