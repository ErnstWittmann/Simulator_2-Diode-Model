import PySpice.Logging.Logging as Logging
import pandas as pd
logger = Logging.setup_logging()
import PySpice.Spice.Simulation
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
import pandas
import numpy as np
from tqdm import tqdm


def calc_MPP(voltage, current):
    p = voltage*current
    pos = np.argmax(p)
    return voltage[pos], current[pos], p[pos]

class Module():
    def __init__(self, circuit, module_parameter_matrix, type= "full-cell_module", index=0,
                 bypass_IS=10e-9, bypass_N=1.0, bypass_RS=1000.0, bypass_BV=50):
        self.circuit = circuit
        self.module_params = module_parameter_matrix
        self.n_strings, self.waver_per_string, _ = module_parameter_matrix.shape
        self.waver_per_string -=1
        self.type = type
        self.index = index
        self.bypass_IS = bypass_IS
        self.bypass_N = bypass_N
        self.bypass_RS = bypass_RS
        self.bypass_BV = bypass_BV

    def add_module(self, end_node):
        if self.type == "full-cell_module":
            for s in range(self.n_strings):
                self.circuit.model(f"BypassDiode_{self.index}_{s}", "D",
                                   IS=self.bypass_IS@u_A, N=self.bypass_N,
                                   RS=self.bypass_RS@u_Ohm, BV=self.bypass_BV@u_V)
                if s + 1 == self.n_strings: end_s = end_node
                else: end_s = f"string_{self.index}_{s + 1}"
                self.circuit.D(f"bD_{self.index}_{s}", end_s, f"string_{self.index}_{s}", model=f"BypassDiode_{self.index}_{s}")
                #print(f"bD_{self.index}_{s} {end_s}, string_{self.index}_{s}")
                self.__add_waver_to_string__(col=s, end_node=end_s)
        else: None #print("there is only full-cell_module given")
        return self.circuit

    def __add_waver_to_string__(self, col, end_node):
        first_waver_of_string = True
        for row in range(self.waver_per_string):
            self.__add_waver__(col, row, self.module_params[col, row], first_waver_of_string)
            first_waver_of_string = False


        self.circuit.R(f"s_{self.index}_{col}_{self.waver_per_string}", f"n_{self.index}_{col}_{self.waver_per_string}",
                  f"w_{self.index}_{col}_{self.waver_per_string}", self.module_params[col, self.waver_per_string, 6] @ u_Ohm)
        #print(f"n_{waver_per_string}_{col} -> w_{waver_per_string}_{col}")

        self.__add_waver_part__(col, self.waver_per_string, f"w_{self.index}_{col}_{self.waver_per_string}",
                       end_node, *self.module_params[col, self.waver_per_string, :-1])

    def __add_waver__(self,  col, row, waver_info, first_waver_of_string=False):
        if first_waver_of_string == True: node_s = f"string_{self.index}_{col}"
        else: node_s = f"n_{self.index}_{col}_{row}"
        self.circuit.R(f"s_{self.index}_{col}_{row}", node_s, f"w_{self.index}_{col}_{row}", waver_info[6] @ u_Ohm)
        #print(f"{node_s} -> w_{self.index}_{col}_{row}")

        # print("R_S_{}_{}".format(n, s))
        self.__add_waver_part__(col, row, f"w_{self.index}_{col}_{row}", f"m_{self.index}_{col}_{row}", *waver_info[:-1])
        #print(f"w_{self.index}_{col}_{row} -> m_{self.index}_{col}_{row}")
        self.circuit.V(f"_{self.index}_{col}_{row}", f"m_{self.index}_{col}_{row}", f"n_{self.index}_{col}_{row + 1}", 0 @ u_V)
        #print(f"m_{self.index}_{col}_{row} -> n_{self.index}_{col}_{row + 1}")

    def __add_waver_part__(self, col, row, start_node, end_node, isc = 0.2@u_A, rp = 1000@u_Ohm,  is_good=1e-9, n_good=1.5, is_bad=1e-9, n_bad=2.0):
        self.circuit.model(f"gD_{self.index}_{col}_{row}", "D", IS=is_good, N=n_good)
        self.circuit.model(f"bD_{self.index}_{col}_{row}", "D", IS=is_bad, N=n_bad)

        self.circuit.D(f"g_{self.index}_{col}_{row}", start_node, end_node, model=f"gD_{self.index}_{col}_{row}")
        self.circuit.D(f"b_{self.index}_{col}_{row}", start_node, end_node, model=f"bD_{self.index}_{col}_{row}")
        self.circuit.R(f"p_{self.index}_{col}_{row}", start_node, end_node, rp)
        self.circuit.I(f"_{self.index}_{col}_{row}", end_node, start_node, isc)

class Waver(SubCircuit):
    __nodes__=("n1","n2")
    def __init__(self, name, rp = 1000@u_Ohm, isc = 0.2@u_A, is_good=1e-9, n_good=1.5, is_bad=1e-9, n_bad=2.0):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.model("goodDiode", "D", IS=is_good, N=n_good)
        self.model("badDiode", "D", IS=is_bad, N=n_bad)

        self.D(1, "n1", "n2", model="goodDiode")
        self.D(2, "n1", "n2", model="badDiode")
        self.R("Parallel", "n1", "n2", rp)
        self.I(1, "n2", "n1", isc)

class Modul(SubCircuit):
    __nodes__ = ("string_1", "n_end")
    def __init__(self, name, waver_list= [], type="full-cell_module"):
        SubCircuit.__init__(self, name, *self.__nodes__)
        n_strings, waver_per_string, _ = waver_list.shape

        for s_id, waver_string in enumerate(waver_list):
            for w_id, waver in enumerate(waver_string):
                # print("W_{}_{}".format(w_id,s_id))
                self.subcircuit(Waver("W_{}_{}".format(w_id,s_id), rp = waver[1], isc = waver[0], is_good=waver[2], n_good=waver[3], is_bad=waver[4], n_bad=waver[5]))

        self.model("Diode", "D", IS=10.352@u_nA, RS=4.6485@u_Ohm, BV=1000@u_V, IBV=0.0001@u_V, N=1)

        if   type == "semi-cell_module":
            if waver_per_string % 2 == 0:
                for s in range(n_strings):
                    if s + 1 == n_strings: self.D("Bypass_{}".format(s), "n_end", f"string_{s + 1}", model="Diode")
                    else: self.D("Bypass_{}".format(s), f"string_{s + 2}", f"string_{s + 1}", model="Diode")

                    #print(f"Diode string_{s + 1} -> string_{s + 2}")

                    self.build_Waver_for_String(1,int(waver_per_string/2), s, n_strings,waver_list)
                    self.build_Waver_for_String(int(waver_per_string / 2)+1,
                                                waver_per_string, s, n_strings,waver_list)

            else: print("number of waver must be equal")

        elif type == "full-cell_module":
            for s in range(n_strings):
                if s+1 == n_strings:
                    self.D("Bypass_{}".format(s), "n_end", f"string_{s+1}", model="Diode")
                    #print(f"Diode string_{s + 1} -> n_end")
                else:
                    self.D("Bypass_{}".format(s), f"string_{s+2}", f"string_{s+1}", model="Diode")
                    #print(f"Diode string_{s+1} -> string_{s+2}")
                self.build_Waver_for_String(1, waver_per_string, s, n_strings, waver_list)

        else: print("Until now you can decide between semi-cell_module and full-cell_module")

    def build_Waver_for_String(self, first_waver, last_waver, s, n_strings, waver_list):
        first_waver_of_string = True
        for w in range(first_waver, last_waver):
            self.buildWaver(w, s, waver_list[s, w - 1, 6] @ u_Ohm, first_waver_of_string)
            first_waver_of_string = False
        self.R("R_S_{}_{}".format(last_waver, s), "n_{}_{}".format(last_waver, s),
               "w_{}_{}".format(last_waver, s), waver_list[s, w, 6])
        #print(f"n_{last_waver}_{s} -> w_{last_waver}_{s}")

        if s + 1 == n_strings:
            self.X("W{}_{}".format(last_waver, s), "W_{}_{}".format(last_waver - 1, s),
                   "w_{}_{}".format(last_waver, s), "n_end")
            #print(f"w_{last_waver}_{s} -> n_end")
        else:
            self.X("W{}_{}".format(last_waver, s), "W_{}_{}".format(last_waver- 1, s),
                   "w_{}_{}".format(last_waver, s), f"string_{s + 2}")
            #print(f"w_{last_waver}_{s} -> string_{s + 2}")

    def buildWaver(self, n, s, rs, first_waver_of_string=False):
        if first_waver_of_string == True:
            self.R("R_S_{}_{}".format(n, s), "string_{}".format(s+1), "w_{}_{}".format(n, s), rs)
            #print(f"string_{s+1} -> w_{n}_{s}")
        else:
            self.R("R_S_{}_{}".format(n, s), "n_{}_{}".format(n, s), "w_{}_{}".format(n, s), rs)
            #print(f"n_{n}_{s} -> w_{n}_{s}")
        #print("R_S_{}_{}".format(n, s))
        self.X("W{}_{}".format(n, s), "W_{}_{}".format(n-1,s), "w_{}_{}".format(n, s), "m_{}_{}".format(n + 1, s))
        #print(f"w_{n}_{s} -> m_{n+1}_{s}")
        self.V("_{}_{}".format(n + 1,s), "m_{}_{}".format(n + 1, s), "n_{}_{}".format(n + 1, s), 0 @ u_V)
        #print(f"m_{n + 1}_{s} -> n_{n + 1}_{s}")

    def explain(self, type):
        if type =="semi-cell_module":
            print("A semi-cell_module has the form: \n"
              "w2-w3 w-w w-w\n"
              "|  | | | | |\n"
              "w1 w4 w w w w\n"
              "|  | | | | |\n"
              "-D1---D---D-\n"
              "|  | | | | |\n"
              "w5 w8 w w w w\n"
              "|  | | | | |\n"
              "w6-w7 w-w w-w")
            print("A Half-Cell Module nowadays has about 120-144 Cells in one cell-string, which are just half as "
                  "big as a the cells within a full-cell module. Furthermore has a Half-Cell 6 strings, but always"
                  "two strings are combined by the same diode.")
        if type == "full-cell_module":
            print("A Full-cell Module has the form: \n"
                  "w10->w20 w11->w21 w12->w22\n"
                  " |    |   |    |   |    |\n"
                  "w00  w30 w01  w31 w02  w32\n"
                  " |    |   |    |   |    |\n"
                  "--D1->-----D2->-----D3->--\n")
            print("A Full-Cell Module nowadays has about 60-72 Cell in one cell-string and contrains 3 cell-strings."
                  "Furthermore each String contains one Diode.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    U_start = 0
    U_end = 40
    resolution = 201
    U = np.linspace(U_start, U_end, resolution + 1)

    # Todo: ##########################################################################
    # make programm faster
    circuit = Circuit("Test")
    circuit.V(1, "string_0_0", circuit.gnd, 1 @ u_V)
    m=Module(circuit, module_parameter_matrix=wl(), type= "full-cell-module", index=0)
    m.add_module("n_5")
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    circuit.V(100, "n_5", circuit.gnd, 0 @ u_V)
    analysis = simulator.dc(V1=slice(U_start, U_end, (U_end - U_start) / resolution))

    U_1 = np.linspace(U_start, U_end, resolution + 1)
    I_1 = np.array(analysis["V100"])
    plt.figure(figsize=(5, 4))
    plt.plot(U_1, -I_1, label="Module fast")
    U_Mpp, I_Mpp, P_Mpp = calc_MPP(U_1, -I_1)
    plt.show()
    print("P_Mpp: ", P_Mpp)

################################################################################################

    circuit = Circuit("Module")
    m = Modul("ModulNorm", waver_list=wl(), type="full-cell_module")
    circuit.subcircuit(m)
    circuit.V(1, "n_1", circuit.gnd, 1 @ u_V)
    circuit.X("Modul1", "ModulNorm", "n_1", "n_5")
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    circuit.V(100, "n_5", circuit.gnd, 0 @ u_V)
    analysis = simulator.dc(V1=slice(U_start, U_end, (U_end - U_start) / resolution))


    U_1 = np.linspace(U_start, U_end, resolution + 1)
    I_1 = np.array(analysis["V100"])
    #plt.figure(figsize=(5,4))
    plt.plot(U_1, -I_1, label="Module with Shadding")
    U_Mpp, I_Mpp, P_Mpp = calc_MPP(U_1, -I_1)
    print("P_Mpp: ", P_Mpp)
    plt.plot(U_Mpp, I_Mpp, "x")
    plt.ylim(0, 6)
    plt.grid()
    plt.xlabel("U/V", fontsize=16)
    plt.ylabel("I/A", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    plt.show()