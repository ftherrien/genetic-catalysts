
from ocdata.vasp import write_vasp_input_files
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
from ocdata.surfaces import Surface
from ocdata.combined import Combined

import logging
import numpy as np
import os
import pickle
import time

class StructureSampler():
    '''
    A class that creates adsorbate/bulk/surface objects and
    writes vasp input files for one of the following options:
    - one random adsorbate/bulk/surface/config, based on a specified random seed
    - one specified adsorbate, n specified bulks, and all possible surfaces and configs
    - one specified adsorbate, n specified bulks, one specified surface, and all possible configs

    The output directory structure will look like the following:
    - For sampling a random structure, the directories will be `random{seed}/surface` and
        `random{seed}/adslab` for the surface alone and the adsorbate+surface, respectively.
    - For enumerating all structures, the directories will be `{adsorbate}_{bulk}_{surface}/surface`
        and `{adsorbate}_{bulk}_{surface}/adslab{config}`, where everything in braces are the
        respective indices.

    Attributes
    ----------
    args : argparse.Namespace
        contains all command line args
    logger : logging.RootLogger
        logging class to print info
    adsorbate : Adsorbate
        the selected adsorbate object
    all_bulks : list
        list of `Bulk` objects
    bulk_indices_list : list
        list of specified bulk indices (ints) that we want to select

    Public methods
    --------------
    run()
        selects the appropriate materials and writes to files
    '''

    def __init__(self, args, catalyst):
        '''
        Set up args from argparse, random seed, and logging.
        '''
        self.args = args

        self.catalyst = catalyst
        
        self.logger = logging.getLogger()
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S')
        self.logger.setLevel(logging.INFO if self.args.verbose else logging.WARNING)

        # if self.args.enumerate_all_structures:
        #     self.bulk_indices_list = [int(ind) for ind in args.bulk_indices.split(',')]
        #     self.logger.info(f'Enumerating all surfaces/configs for adsorbate {self.args.adsorbate_index} and bulks {self.bulk_indices_list}')
        # else:
        #     self.logger.info('Sampling one random structure')
        #     np.random.seed(self.args.seed)

    def run(self):
        '''
        Runs the entire job: generates adsorbate/bulk/surface objects and writes to files.
        '''
        start = time.time()

        if self.args.enumerate_all_structures:
            t_ads = time.time()
            self.adsorbate = Adsorbate(self.args.adsorbate_db, self.args.adsorbate_index)
            t_ads = time.time() - t_ads
            
        t_bulks = time.time()
        self._load_bulks()
        t_bulks = time.time() - t_bulks
        
        t_load = time.time()
        res, timing_load = self._load_and_write_surfaces()
        t_load = time.time() - t_load

        timing = [t_ads, t_bulks, t_load, timing_load]

        return res, timing, os.path.join(self.args.output_dir, self.output_name_template)

        end = time.time()
        self.logger.info(f'Done! ({round(end - start, 2)}s)')

    def _load_bulks(self):
        '''
        Loads bulk structures (one random or a list of specified ones)
        and stores them in self.all_bulks
        '''
        self.all_bulks = []
        with open(self.args.bulk_db, 'rb') as f:
            bulk_db_lookup = pickle.load(f)

        if self.args.enumerate_all_structures:
            #for ind in self.bulk_indices_list:
            self.all_bulks.append(Bulk(bulk_db_lookup, self.args.precomputed_structures, self.catalyst.bulk_index))
        else:
            self.all_bulks.append(Bulk(bulk_db_lookup, self.args.precomputed_structures))

    def _load_and_write_surfaces(self):

        timing_load = []

        '''
        Loops through all bulks and chooses one random or all possible surfaces;
        writes info for that surface and combined surface+adsorbate
        '''
        for bulk_ind, bulk in enumerate(self.all_bulks):
            # possible_surfaces = bulk.get_possible_surfaces()
            t = time.time()
            possible_surfaces, timing_enum = bulk.enumerate_surfaces(self.catalyst.miller)
            print(possible_surfaces)
            timing_load.append(time.time() - t)
            timing_load.append(timing_enum)

            surface_info = possible_surfaces[self.catalyst.term%len(possible_surfaces)]

            surface = Surface(bulk, surface_info, self.catalyst.term%len(possible_surfaces), len(possible_surfaces))
            timing_load.append(surface.timing)
            
            t = time.time()
            out =  self._combine_and_write(surface, self.catalyst.bulk_index, self.catalyst.term%len(possible_surfaces))
            timing_load.append(time.time() - t)

            return out, timing_load

            
            
            # if self.args.enumerate_all_structures:
            #     if self.args.surface_index is not None:
            #         assert 0 <= self.args.surface_index < len(possible_surfaces), 'Invalid surface index provided'
            #         self.logger.info(f'Loading only surface {self.args.surface_index} for bulk {self.bulk_indices_list[bulk_ind]}')
            #         included_surface_indices = [self.args.surface_index]
            #     else:
            #         self.logger.info(f'Enumerating all {len(possible_surfaces)} surfaces for bulk {self.bulk_indices_list[bulk_ind]}')
            #         included_surface_indices = range(len(possible_surfaces))

            #     for cur_surface_ind in included_surface_indices:
            #         surface_info = possible_surfaces[cur_surface_ind]
            #         surface = Surface(bulk, surface_info, cur_surface_ind, len(possible_surfaces))
            #         self._combine_and_write(surface, self.bulk_indices_list[bulk_ind], cur_surface_ind)
            # else:
            #     surface_info_index = np.random.choice(len(possible_surfaces))
            #     surface = Surface(bulk, possible_surfaces[surface_info_index], surface_info_index, len(possible_surfaces))
            #     self.adsorbate = Adsorbate(self.args.adsorbate_db)
            #     self._combine_and_write(surface)


    def _combine_and_write(self, surface, cur_bulk_index=None, cur_surface_index=None):
        '''
        Add the adsorbate onto a given surface in a Combined object.
        Writes output files for the surface itself and the combined surface+adsorbate

        Args:
            surface: a Surface object to combine with self.adsorbate
            cur_bulk_index: current bulk index from self.bulk_indices_list
            cur_surface_index: current surface index if enumerating all
        '''
        if self.args.enumerate_all_structures:
            self.output_name_template = f'{self.args.adsorbate_index}_{cur_bulk_index}_{"-".join([str(b) for b in self.catalyst.miller])}_{self.catalyst.term}_{self.catalyst.site}'
        else:
            self.output_name_template = f'random{self.args.seed}'

        t = time.time()
        self._write_surface(surface, self.output_name_template)
        print("Write surface time", time.time() - t)

        t = time.time()
        combined = Combined(self.adsorbate, surface, self.args.enumerate_all_structures, self.catalyst.site)
        print("Combined", time.time() - t)
        
        return self._write_adsorbed_surface(combined, self.output_name_template)

    def _write_surface(self, surface, output_name_template):
        '''
        Write VASP input files and metadata for the surface alone.

        Args:
            surface: the Surface object to write info for
            output_name_template: parent directory name for output files
        '''
        bulk_dict = surface.get_bulk_dict()
        bulk_dir = os.path.join(self.args.output_dir, output_name_template, 'surface')
        write_vasp_input_files(bulk_dict['bulk_atomsobject'], bulk_dir)
        self._write_metadata_pkl(bulk_dict, os.path.join(bulk_dir, 'metadata.pkl'))
        self.logger.info(f"wrote surface ({bulk_dict['bulk_samplingstr']}) to {bulk_dir}")

    def _write_adsorbed_surface(self, combined, output_name_template):
        '''
        Write VASP input files and metadata for the adsorbate placed on surface.

        Args:
            combined: the Combined object to write info for, containing any number of adslabs
            output_name_template: parent directory name for output files
        '''
        self.logger.info(f'Writing {combined.num_configs} adslab configs')
        # for config_ind in range(combined.num_configs):
        #     if self.args.enumerate_all_structures:
        #         adsorbed_bulk_dir = os.path.join(self.args.output_dir, output_name_template, f'adslab{config_ind}')
        #     else:
        #         adsorbed_bulk_dir = os.path.join(self.args.output_dir, output_name_template, 'adslab')
        #     adsorbed_bulk_dict = combined.get_adsorbed_bulk_dict(config_ind)
        #     write_vasp_input_files(adsorbed_bulk_dict['adsorbed_bulk_atomsobject'], adsorbed_bulk_dir)
        #     self._write_metadata_pkl(adsorbed_bulk_dict, os.path.join(adsorbed_bulk_dir, 'metadata.pkl'))
        #     if config_ind == 0:
        #         self.logger.info(f"wrote adsorbed surface ({adsorbed_bulk_dict['adsorbed_bulk_samplingstr']}) to {adsorbed_bulk_dir}")

        adsorbed_bulk_dir = os.path.join(self.args.output_dir, output_name_template, f'adslab{0}')
        adsorbed_bulk_dict = combined.get_adsorbed_bulk_dict(0)
        write_vasp_input_files(adsorbed_bulk_dict['adsorbed_bulk_atomsobject'], adsorbed_bulk_dir)
        self._write_metadata_pkl(adsorbed_bulk_dict, os.path.join(adsorbed_bulk_dir, 'metadata.pkl'))
        # if config_ind == 0:
        #     self.logger.info(f"wrote adsorbed surface ({adsorbed_bulk_dict['adsorbed_bulk_samplingstr']}) to {adsorbed_bulk_dir}")
        return adsorbed_bulk_dict['adsorbed_bulk_atomsobject']
            

    def _write_metadata_pkl(self, dict_to_write, path):
        '''
        Writes a dict as a metadata pickle

        Args:
            dict_to_write: dict containing all info to dump as file
            path: output file path
        '''
        file_path = os.path.join(path, 'metadata.pkl')
        with open(path, 'wb') as f:
            pickle.dump(dict_to_write, f)

