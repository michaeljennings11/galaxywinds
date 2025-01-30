# run galaxywinds
import errno
import os
import subprocess

import numpy as np

from galaxywinds import FB20, clouds, config, constants, utils


def run(executable="colt", config_file="config.yaml", build=True, tasks=1, threads=8):
    """Run an executable."""
    executable = config.colt_dir + "/" + executable
    if build:
        utils.silentremove(executable)
        subprocess.call(["python", "build.py"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    # os.environ['OMP_NUM_THREADS'] = str(threads)
    subprocess.call(["mpirun", "-n", str(tasks), executable, config_file])


if __name__ == "__main__":
    print("Running galaxy_winds now!")

    print("Running FB20 for wind solution...")
    wind_solution = FB20.run_FB20()
    print("Finished FB20!")

    print(f"Running genclouds...")
    config_files_list = clouds.generate_clouds(wind_solution)
    print("Finished saving cloud datacubes and colt config files!")

    print("Running COLT ionization...")
    for i, config_file in enumerate(config_files_list):
        run(config_file=config_file, build=False)
