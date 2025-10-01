# Global config file
# Change directory paths accordingly

import os

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, os.pardir))

# colt directory path
colt_dir = "/Users/mjennings/winds"
colt_executable = "colt_uvb2"

# bpass directory path
bpass_dir = "/Users/mjennings/Projects/galaxy_winds/data/external/SED/bpass-imf135_100"

# colt inputs directory path
model_outs_dir = "/Users/mjennings/Projects/galaxy_winds/data/interim/outputs"
colt_cubes_dir = "/Users/mjennings/Projects/galaxy_winds/data/interim/enzo_to_colt"
colt_out_dir = "/Users/mjennings/Projects/galaxy_winds/data/interim/colt_outputs"
ion_config_dir = colt_cubes_dir + "/config_files"

# cooling table file
cooling_file = "/Users/mjennings/Projects/MultiphaseGalacticWind/CoolingTables/z_0.000.hdf5"  # From Wiersma et al. (2009) appropriate for z=0 UVB

# model name
model = "CLASSY_J0036-3333_UVBonly_prolate"

# wind param file
wind_param_file = PROJECT_ROOT + "/wind_params_CLASSY_J0036-3333.yaml"

# cloud param file
cloud_param_file = PROJECT_ROOT + "/cloud_params.yaml"

# runtime options
run_ionization_module = True
run_line_module = True
