# Global config file
# Change directory paths accordingly

import os

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, os.pardir))

# colt directory path
colt_dir = "/Users/mjennings/colt"

# bpass directory path
bpass_dir = "/Users/mjennings/Projects/galaxy_winds/data/external/SED/bpass-imf135_100"

# colt inputs directory path
colt_cubes_dir = "/Users/mjennings/Projects/galaxy_winds/data/interim/enzo_to_colt"
ion_config_dir = colt_cubes_dir + "/config_files"

# cooling table file
cooling_file = "/Users/mjennings/Projects/MultiphaseGalacticWind/CoolingTables/z_0.000.hdf5"  # From Wiersma et al. (2009) appropriate for z=0 UVB

# wind param file
wind_param_file = PROJECT_ROOT + "/wind_params.yaml"

# cloud param file
cloud_param_file = PROJECT_ROOT + "/cloud_params.yaml"
