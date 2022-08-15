import numpy as np
from catalystid import catalystID, argsObject

args = argsObject()

args.checkpoint_path = "painn_h1024_bs4x8_is2re_all.pt"

args.config_yml_path = "/home/e/esargent/felixt/softwares/ocp/configs/is2re/all/painn/painn_h1024_bs8x4.yml"

args.bulk_db = "slabs_new.pkl"

args.adsorbate_db = "adsorbates_new.pkl"

args.output_dir = "best_structures"

args.adsorbate_index = 5 # CO

cat = catalystID(id=31, miller=[-11,1,1], term = 14, site = 9,args=args)

print("Catalyst:", cat)
print("Value:", cat.value)
