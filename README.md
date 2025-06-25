1-10POSCAR, 1-24POSCAR, 1-83POSCAR, pure_water-POSCAR and param.json are the initial structure files for the DP-Gen iteration cycle.

input.json is the input file required to train the standard DP model.

dp-long.pb is a model trained using the standard DP method after DP-Gen conformation screening and collection.

atomic dipole1-83.npy, box1-83.npy, coord1-83.npy and dw.json are the input files required to train the Deep Dipole model.

dw-1000w.pb is the trained Deep Dipole model.

ener.json and dw-1000w.pb are the input files required to train the DPLR model.

in.data and input.lammps are the input files to run DPLR simulation.

Ca-0-coordination.py, ava-angle-dist-error bar line.py, hbnum.py, adf-make.py, rdf-plot.py and stderr-water-test-new.py are script files for data analysis.

note: The DPLR model file is too large to upload. If you need it, please contact the corresponding author to obtain it.
