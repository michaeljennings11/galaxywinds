# run galaxywinds
import numpy as np

from galaxywinds import FB20, clouds, constants, utils

if __name__ == "__main__":
    print("Running galaxy_winds now!")

    print("Running FB20 for wind solution...")
    wind_solution = FB20.run_FB20()
    print("Finished FB20!")

    print(f"Running genclouds...")
    clouds.generate_clouds(wind_solution)
    print("Finished saving cloud datacubes and colt config files!")
