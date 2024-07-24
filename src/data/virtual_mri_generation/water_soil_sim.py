""" 
Benchmark M1.2 static root system in soil (root hydrualics with Meunier and the classic sink)

also works parallel with mpiexec (slower, due to overhead?)
"""

import os
import sys

with open("../../DUMUX_path.txt", "r") as file:
    DUMUX_path = file.read()

sys.path.append("f{DUMUX_path}/CPlantBox/")
sys.path.append(f"{DUMUX_path}/CPlantBox/src/")
sys.path.append(
    f"{DUMUX_path}/dumux-rosi/build-cmake/cpp/python_binding/"
)  # dumux python binding
sys.path.append(f"{DUMUX_path}/dumux-rosi/python/modules/")  # python wrappers

import plantbox as pb
from functional.xylem_flux import XylemFluxPython  # Python hybrid solver
import functional.van_genuchten as vg
from functional.root_conductivities import *
import rsml.rsml_reader as rsml
import visualisation.vtk_plot as vp

from rosi_richards import RichardsUG  # C++ part (Dumux binding)
from richards import RichardsWrapper  # Python part
from rhizo_models import plot_transpiration

from math import *
import numpy as np
import matplotlib.pyplot as plt
import timeit
from mpi4py import MPI
import xml.etree.ElementTree as ET

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def sinusoidal(t):
    return np.sin(2.0 * pi * np.array(t) - 0.5 * pi) + 1.0


class SoilWaterSimulation:
    def __init__(
        self,
        mesh_path,
        rsml_path,
        output_path,
        soil_type,
        initial=-659.8 + 10,
        trans=6.4,
        wilting_point=-15000,
        age_dependent=False,
        sim_time=1,
    ):
        """
        Simulates the water movement in the soil for a given root system and soil properties.

        Args:
        - mesh_path (str): Path to the mesh file, for how fine the grid should be for which the water
                            movement will be simulated.
        - rsml_path (str): Path to the rsml file, containing the root system.
        - output_path (str): Path to the folder where the output will be saved.
        - root_system: Root system simulation object.
        - soil_type (str): Type of ground ("sand", "loam", "clay").
        - initial (float): Initial pressure head. Default is -659.8 + 10.
        - trans: Transpiration rate (Amount of water which is released from the plant). Default is 6.4.
        - wilting_point (float): Wilting point. Default is -15000.
        - sim_time (float): Simulation time. Default is 7.1.
        - age_dependent (bool): If True, the root conductivities will be age dependent. Default is False.
        """

        self.mesh_path = mesh_path
        self.rsml_path = rsml_path
        self.output_path = output_path
        self._set_ground_type(soil_type)
        self.soil_type = soil_type
        self.initial = initial
        self.trans = trans
        self.wilting_point = wilting_point
        self.sim_time = sim_time
        self.age_dependent = age_dependent

        if wilting_point is None:
            if self.soil_type == "sand":
                self.wilting_point = -2500
            elif self.soil_type == "loam":
                self.wilting_point = -4000
            elif self.soil_type == "clay":
                self.wilting_point = -15000

    def _sinusoidal(self, t) -> float:
        """
        Calculates the value of a sinusoidal function for a given time.

        Args:
        - t (float): Time.

        Returns:
        - float: Value of the sinusoidal function for the given time.
        """
        return np.sin(2.0 * pi * np.array(t) - 0.5 * pi) + 1.0

    def _set_ground_type(self, soil_type):
        """
        Sets the ground type for the soil water simulation.
        """
        sand = [0.045, 0.43, 0.15, 3, 1000]
        loam = [0.08, 0.43, 0.04, 1.6, 50]
        clay = [0.1, 0.4, 0.01, 1.1, 10]

        if soil_type == "sand":
            self.soil_type_params = sand
            self.dt = 120
        elif soil_type == "loam":
            self.soil_type_params = loam
            self.dt = 360
        elif soil_type == "clay":
            self.soil_type_params = clay
            self.dt = 360

    def _init_soil_model(self):
        sp = vg.Parameters(self.soil_type_params)  # for debugging
        self.s = RichardsWrapper(RichardsUG())
        self.s.initialize()
        self.s.readGrid(self.mesh_path)  # [cm]
        self.s.setHomogeneousIC(self.initial, True)  # cm pressure head, equilibrium
        self.s.setTopBC("noFlux")
        self.s.setBotBC("noFlux")
        self.s.setVGParameters([self.soil_type_params])
        self.s.setParameter("Newton.EnableChop", "True")
        self.s.setParameter("Newton.EnableAbsoluteResidualCriterion", "True")
        self.s.setParameter(
            "Soil.SourceSlope", "1000"
        )  # turns regularisation of the source term on, will change the shape of actual transpiration...
        self.s.initializeProblem()
        self.s.setCriticalPressure(self.wilting_point)

    def _init_xylem_model(self):
        self.r = XylemFluxPython(self.rsml_path)
        init_conductivities(self.r, self.age_dependent)

    def _print_progress_bar(self, iteration, total, info="", bar_length=50):
        progress = iteration / total
        arrow = "=" * int(round(progress * bar_length) - 1) + ">"
        spaces = " " * (bar_length - len(arrow))

        sys.stdout.write(f"\rProgress: [{arrow + spaces}] {int(progress*100)}% {info}")
        sys.stdout.flush()  # This is important to ensure the progress is updated

    def _add_water_content(self, vtu_path):
        """
        Adds the water content to the soil.
        """
        water_content = self.s.getWaterContent()

        tree = ET.parse(vtu_path)
        root = tree.getroot()

        cell_data = root.find(".//CellData")

        xml_elem = ET.SubElement(
            cell_data,
            "DataArray",
            attrib={
                "type": "Float32",
                "Name": "water content",
                "NumberOfComponents": "1",
                "format": "ascii",
            },
        )

        # Example water content data (as a string)
        water_content_data = " ".join(str(num) for num in water_content.ravel())

        # Add the water content data as text in the DataArray element
        xml_elem.text = water_content_data

        tree.write(vtu_path)

    def run(self) -> str:
        """
        Runs the soil water simulation for the given root system and soil properties.

        Returns:
        - filename (str): Name of the vtu file, containing the soil water simulation data.
        """
        self._init_soil_model()
        self._init_xylem_model()

        """ Coupling (map indices) """
        picker = lambda x, y, z: self.s.pick([x, y, z])
        cci = picker(
            self.r.rs.nodes[0].x, self.r.rs.nodes[0].y, self.r.rs.nodes[0].z
        )  # collar cell index
        self.r.rs.setSoilGrid(picker)  # maps segment

        """ sanity checks """
        # r.plot_conductivities()
        self.r.test()  # sanity checks
        rs_age = np.max(self.r.get_ages())
        # print("press any key"); input()

        # os.chdir(self.output_path)

        """ Numerical solution (a) """
        start_time = timeit.default_timer()
        x_, y_, z_ = [], [], []
        sink1d = []
        sx = self.s.getSolutionHead()  # inital condition, solverbase.py
        dt = self.dt / (
            24 * 3600
        )  # seconds divided by rest (don't change rest) # 120 sand, 360 loam, clay # coupling zu groß wenns oszilliert # sand -100 , 15sec
        skip = 1

        N = round(self.sim_time / dt)
        t = 0.0
        run_number = 0
        for i in range(0, N):
            if rank == 0:  # Root part is not parallel
                rx = self.r.solve(
                    rs_age + t,
                    -self.trans * sinusoidal(t),
                    0.0,
                    sx,
                    True,
                    self.wilting_point,
                    [],
                )  # xylem_flux.py, cells = True
                fluxes = self.r.soilFluxes(
                    rs_age + t, rx, sx, False
                )  # class XylemFlux is defined in MappedOrganism.h, approx = True

            else:
                fluxes = None

            fluxes = comm.bcast(fluxes, root=0)  # Soil part runs parallel

            water = self.s.getWaterVolume()
            self.s.setSource(fluxes.copy())  # richards.py
            self.s.solve(dt)
            old_sx = sx.copy()
            sx = self.s.getSolutionHead()  # richards.py
            soil_water = (self.s.getWaterVolume() - water) / dt

            if rank == 0 and i % skip == 0:
                min_sx = np.min(sx)
                min_rx = np.min(rx)
                max_sx = np.max(sx)
                max_rx = np.max(rx)
                x_.append(t)
                sum_flux = 0.0
                for f in fluxes.values():
                    sum_flux += f
                # print("Summed fluxes ", sum_flux, "= collar flux", self.r.collar_flux(rs_age + t, rx, sx), "= prescribed", -self.trans * sinusoidal(t))

                y_.append(soil_water)  # cm3/day (soil uptake)
                # z_.append(sum_flux)  # cm3/day (root system uptake)
                z_.append(float(self.r.collar_flux(rs_age + t, rx, sx)))  # cm3/day

                # collar_flux = round(self.r.collar_flux(rs_age + t, rx, sx)[0], 3)
                prescribed = round(-self.trans * sinusoidal(t), 3)
                self._print_progress_bar(
                    i + 1,
                    N,
                    # info=f"collar flux: {collar_flux} | {prescribed} :prescribed",
                )

                # n = round(float(i) / float(N) * 100.)
                # print("[" + ''.join(["*"]) * n + ''.join([" "]) * (100 - n) + "], soil [{:g}, {:g}] cm, root [{:g}, {:g}] cm, {:g} days {:g}\n"
                #     .format(min_sx, max_sx, min_rx, max_rx, self.s.simTime, rx[0]))

                run_number += 1

            t += dt

        print(
            "\n"
            + "\033[92m"
            + "===================================================="  # Green text
            + "\n"
            + "||       Water Soil Simulation: COMPLETE!        ||"
            + "\n"
            + "===================================================="
            + "\033[0m"
        )  # Reset text color

        rsml_name = self.rsml_path[
            self.rsml_path.rfind("/") + 1 : self.rsml_path.rfind(".")
        ]
        self.s.writeDumuxVTK(rsml_name)

        # vp.write_soil(
        #     rsml_name,
        #     self.s,
        #     (-5, -5, -5),
        #     (5, 5, 5),
        #     (20, 2, 21),
        # )

        filename = (
            str(rsml_name)
            + "_soil_"
            + self.soil_type
            + "_initial"
            + str(self.initial)
            + "_sim-time"
            + str(self.sim_time)
            + ".vtu"
        )
        new_filename = self.output_path + "/" + filename
        os.system("mv " + str(rsml_name) + "-00000.vtu " + new_filename)
        self._add_water_content(new_filename)

        print(f"{rsml_name}.pvd")
        os.remove(f"{rsml_name}.pvd")

        return filename


# # Example usage:
# my_soil_sim = SoilWaterSimulation(
#     "cylinder_r_0.032_d_-0.22_res_0.01_testing_fast.msh",
#     "Bench_lupin_day_10.rsml",
#     "./",
#     "loam",
# )
# my_soil_sim.run()
