from dbspace.signal.dLFP.impedances import Z_class
import matplotlib.pyplot as plt


print("Doing Impedance Mismatch Analysis - Script")
Z_lib = Z_class(
    impedance_data_path="/home/vscode/data/Anatomy/CT/Zs_GMs_Es.xlsx",
    pts=["DBS901", "DBS903", "DBS905", "DBS906", "DBS907", "DBS908"],
)
Z_lib.get_recZs()
Z_lib.plot_recZs()
Z_lib.dynamics_measures()
