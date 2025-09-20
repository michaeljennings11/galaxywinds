# run galaxywinds

import os
import subprocess

import numpy as np

from galaxywinds import FB20, clouds, config, constants, utils

model_outs_dir = config.model_outs_dir
model = config.model
model_dir = os.path.join(model_outs_dir, model)
data_cube_dir = os.path.join(model_dir, "data_cubes")
config_files_dir = os.path.join(model_dir, "config_files")
ionization_dir = os.path.join(model_dir, "ionization")
line_dir = os.path.join(model_dir, "line")


def run(executable="colt", config_file="config.yaml", build=True, tasks=1, threads=8):
    """Run an executable."""
    executable = config.colt_dir + "/" + executable
    print(f"Running executable: {executable}.")
    if build:
        print(f"Build?: {build}")
        utils.silentremove(executable)
        subprocess.call(["python", config.colt_dir + "/build.py"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    # os.environ['OMP_NUM_THREADS'] = str(threads)
    subprocess.call(["mpirun", "-n", str(tasks), executable, config_file])


if __name__ == "__main__":
    print("Running galaxy_winds now!")

    # create model output directory structure
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_cube_dir, exist_ok=True)
    os.makedirs(config_files_dir, exist_ok=True)
    os.makedirs(config_files_dir + "/ion_configs", exist_ok=True)
    os.makedirs(config_files_dir + "/line_configs", exist_ok=True)
    os.makedirs(ionization_dir, exist_ok=True)
    os.makedirs(line_dir, exist_ok=True)

    print("Running FB20 for wind solution...")
    wind_solution = FB20.run_FB20()
    print("Finished FB20!")

    print(f"Running genclouds...")
    ion_config_files_list, line_config_files_list = clouds.generate_clouds(
        wind_solution
    )
    print("Finished saving cloud datacubes and colt config files!")

    if config.run_ionization_module:
        print("Running COLT ionization...")
        for i, ion_config_file in enumerate(ion_config_files_list):
            print(f"Starting cloud {i}.")
            run(config_file=ion_config_file, build=False)
            print(f"Finished cloud {i}!")
        print("Finished running ionization!")

    # print("Generating line spectrum config files...")
    # line_config_files_list = spectrum.generate_line_configs(ion_config_files_list)

    if config.run_line_module:
        print("Running COLT line mcrt...")
        for i, line_config_file in enumerate(line_config_files_list):
            run(config_file=line_config_file, build=False)
        print("Finished running line mcrt!")
