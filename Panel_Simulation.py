from Module import Module
import numpy as np

import pandas as pd

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from tqdm import tqdm
import time as time_lib

class Panel_Simulator():
    def __init__(self, module_type="full-cell_module"):

        self.module_type = module_type

    def get_module_parameter_matrix(self, G):
        """
        -früher create_module_matrix
        Uses the information of "module_id"<->"LTSpice_Module_Parameter.csv" to create a module with no defects.
        Waver_list will be used to create the UI-Curve
        :param G: sun irradiance (e.g. ghi)
        :return: waver list
        """
        waver_list = np.ones([self.n_strings_per_modul, self.n_waver_per_string, 7])
        waver_list[:, :, 0] = self.module_ISC/1000*G
        waver_list[:, :, 1] = self.module_shunt_resistance
        waver_list[:, :, 2] = self.module_good_diode_reverse_leakage_current
        waver_list[:, :, 3] = self.module_good_diode_N
        waver_list[:, :, 4] = self.module_bad_diode_reverse_leakage_current
        waver_list[:, :, 5] = self.module_bad_diode_N
        waver_list[:, :, 6] = self.module_serial_resistance
        return waver_list

    def get_panel_parameter_matrix(self, G, n_parallel_strings=1, n_moduls_per_string=5):
        """
        -create_panel_parameter_matrix-
        Uses the information of "module_id"<->"LTSpice_Module_Parameter.csv" to create a panel with modules with no defects.
        :param G: sun irradiance (e.g. ghi)
        :return: waver list
        """
        module_param = self.get_module_parameter_matrix(G)
        modul_param_string = []
        for n in range(n_moduls_per_string):  # combine modules to strings
            modul_param_string.append(module_param)
        panel_list = []
        for n in range(n_parallel_strings):  # combine strings to panel
            panel_list.append(modul_param_string)
        panel_list = np.array(panel_list)
        return panel_list

    #########################################################################
    def create_module(self, module_parameter_matrix, index = 0, bypass_IS=10e-9, bypass_N=1.0, bypass_RS=1000.0, bypass_BV=50):
        """For faster calculation.
        This methode creates a module, that parameters can be changed by "update module",
        futhermore it can be simulated with "simulate module". """
        circuit = Circuit("Module")
        circuit.V(1, "string_0_0", circuit.gnd, 1 @ u_V)
        Module(circuit, module_parameter_matrix, self.module_type, index, bypass_IS, bypass_N, bypass_RS, bypass_BV).add_module("n_5")
        circuit.V(100, "n_5", circuit.gnd, 0 @ u_V)
        self.circuit=circuit
        return self.circuit

    def create_panel(self, panel_parameter_matrix):
        """
        IMPORTANT - This Panal analysis is very accurate but can take a lot of time for calculation.
        This methode creates a panel, that parameters can be changed by "update panel",
        futhermore it can be simulated with "simulate panel". """
        circuit = Circuit("Panel")
        circuit.V(1, f"string_[0_0]_0", circuit.gnd, 1 @ u_V)
        # COMBINE MODULES TO PARALLEL STRINGS
        for p, parallel_string in enumerate(panel_parameter_matrix):
            for s, module_param in enumerate(parallel_string):
                Module(circuit, module_param, self.module_type, index=f"[{p}_{s}]").add_module(f"string_[{p}_{s+1}]_0")
            circuit.V(f"_string_{p}", f"string_[{p}_{s+1}]_0", "panel_node", 0 @ u_V)
        circuit.V("_panel", "panel_node", circuit.gnd, 0 @ u_V)
        self.circuit = circuit
        return circuit

    ##########################################################################
    def update_module_irradiation(self, module_parameter_matrix, circuit=None):
        """Changes the module irradiation.
        module_parameter_matrix: matrix with all module parameter
        circuit: has to be used, if two models are calculated at the same time"""
        if circuit==None: circuit=self.circuit
        n_strings, waver_per_string, _ = module_parameter_matrix.shape
        for col in range(n_strings):
            for row in range(waver_per_string):
                circuit[f"I_0_{col}_{row}"].dc_value = module_parameter_matrix[col, row, 0]
        self.circuit = circuit
        return self.circuit

    def update_panel_irradiation(self, panel_parameter_matrix, circuit=None):
        """
        IMPORTANT - This Panal analysis is very accurate but can take a lot of time for calculation.
        Changes the module irradiation.
                module_parameter_matrix: matrix with all module parameter
                circuit: has to be used, if two models are calculated at the same time"""
        if circuit == None: circuit = self.circuit
        for p, parallel_string in enumerate(panel_parameter_matrix):
            for s, module_param in enumerate(parallel_string):
                for col, string_param in enumerate(module_param):
                    for row,waver_param in enumerate(string_param):
                        circuit[f"I_[{p}_{s}]_{col}_{row}"].dc_value = waver_param[0]
        self.circuit = circuit
        return self.circuit

    ##########################################################################
    def analyse_module(self, U_start = 0, U_end = 100, resolution=201, temperature=25, nominal_temperature=25, circuit=None):
        """
        calculates the IV_Curve of a module.
        With U_start, U_end and resolution the calculation time can be changed
        :param U_start: minimum Voltage where IV-Curve is calculated
        :param U_end: maximum Voltage where IV_Curve is calculated
        :param resolution: number of calculated data points of for the IV_Curve
        :param temperature:
        :param nominal_temperature:
        :param circuit: circuit of the module, important if two circuits are calculated at same time
        :return: [np.array for Voltage, np.array for Current]
        """
        if circuit == None: circuit = self.circuit
        simulator = circuit.simulator(temperature=temperature, nominal_temperature=nominal_temperature)
        analysis = simulator.dc(V1=slice(U_start, U_end, (U_end-U_start)/resolution))
        self.I_module = -np.array(analysis["V100"])
        self.U_module = np.linspace(U_start, U_end, len(self.I_module))
        return self.U_module, self.I_module

    def analyse_panel(self, ppm_shape =[0,0,0,0,0], U_start = 0, U_end = 100, resolution=201, temperature=25, nominal_temperature=25,  circuit = None):
        """
        IMPORTANT - This Panal analysis is very accurate but can take a lot of time for calculation.
                calculates the IV_Curve of a panel.
                With U_start, U_end and resolution the calculation time can be changed
                :param ppm_shape: the shape of the panel_parameter_matrix
                :param U_start: minimum Voltage where IV-Curve is calculated
                :param U_end: maximum Voltage where IV_Curve is calculated
                :param resolution: number of calculated data points of for the IV_Curve
                :param temperature:
                :param nominal_temperature:
                :param circuit: circuit of the module, important if two circuits are calculated at same time
                :return: [np.array for Voltage, np.array for Current]
                """
        if circuit == None: circuit = self.circuit

        simulator = circuit.simulator(temperature=temperature, nominal_temperature=nominal_temperature)
        analysis = simulator.dc(V1=slice(U_start, U_end, (U_end - U_start) / resolution))
        # READ PANEL INFOS
        self.panel_current = -np.array(analysis["V_panel"])
        res = len(self.panel_current)
        self.U_panel = np.linspace(U_start, U_end, res)

        n_parallel_strings,n_modules_per_string,_,_,_ = ppm_shape
        self.string_currents = np.zeros((n_parallel_strings, res))
        self.module_currents = np.zeros((n_parallel_strings, n_modules_per_string, res))
        for p in range(n_parallel_strings):
            self.string_currents[p] = -np.array(analysis[f"V_string_{p}"])

        return self.U_panel, self.string_currents, self.panel_current

    def analyse_combinedModules(self, iv_curve_list=[]):
        """
        Takes a list of calculated IV_curves of modules, and gives a IV-Curve of a string
        :param iv_curve_list: [[voltage, current], [voltage,current], ... ]
        :return:
        """
        merged_iv_curve = pd.DataFrame()
        merged_iv_curve["I"] = iv_curve_list[0][1]
        merged_iv_curve["U"] = iv_curve_list[0][0]
        merged_iv_curve = merged_iv_curve.sort_values("I")

        for iv_curve in iv_curve_list[1:]:
            mic = pd.DataFrame()
            mic["I"] = iv_curve[1]
            mic["U_ADD"] = iv_curve[0]
            mic = mic.sort_values("I")

            merged_iv_curve = pd.merge_asof(merged_iv_curve, mic, on="I")
            merged_iv_curve["U"] = merged_iv_curve["U"].values + merged_iv_curve["U_ADD"].values
            merged_iv_curve = merged_iv_curve.drop(columns=['U_ADD'])
        merged_iv_curve = merged_iv_curve.dropna()
        P = merged_iv_curve["I"].values * merged_iv_curve["U"].values

        mpp_index = np.argmax(P)
        i_mpp, u_mpp = merged_iv_curve["I"].values[mpp_index], merged_iv_curve["U"].values[mpp_index]

        return u_mpp, i_mpp

    ##########################################################################
    def calc_MPP(self, voltage, current):
        """
        Takes an IV-Curve and calculates MPP of it
        :param voltage: np.array
        :param current: np.array
        :return: V_MPP, I_MPP, P_MPP
        """
        p = voltage * current
        pos = np.argmax(p)
        return voltage[pos], current[pos], p[pos]