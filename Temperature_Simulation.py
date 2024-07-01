import numpy as np

class Temperature():
    def __init__(self, width=60, length=30):
        self.area = width*length
        self.area_m = self.area/10000
        self.absorbtion_factor = 0.005
        self.w = 4190
        self.M = 20
        self.frequenz = 10

    def temperature(self,temp,dtemp=28, irradiation=800):
        return temp+dtemp+irradiation/1000

    def stefan_boltzmann(self, irradiation):
        o = 5.67e-8
        return (irradiation/o*self.area_m*self.absorbtion_factor)**(1/4)*0.8

    def heating(self, irradiation, T_0):
        T = [T_0]
        for T_sb in self.stefan_boltzmann(irradiation):
            T_new = self.w*self.frequenz*60*(T_sb-T[-1])
            print(T_new)
            T.append(T[-1]+T_new)
        return T[1:]

if __name__=="__main__":
    from Simulator import Simulator
    import matplotlib.pyplot as plt
    import pandas as pd

    d = pd.read_csv(r"C:\Users\wittmanne\Desktop\PycharmProjects\GETTING_MODULE_DATA/ERLANGEN/ERLANGEN_COMBINED_DATA/module_string.csv")
    d["date"] = pd.to_datetime(d["date"])
    d["time"] = pd.to_datetime(d["time"])
    d = d.loc[(d["date"] == pd.to_datetime("2021-07-19"))&(d["module_temperature"]>-100)&
              (d["module_id"] == np.unique(d["module_id"].values)[0])]
    print(d.columns)

    sim = Simulator(start_date="2021-07-19", end_date="2021-07-20", frequence=5)
    time, irradiation = sim.simulate_sun()

    plt.plot(time,irradiation**(1/2)+10)
    plt.plot(d["time"].values, d["module_temperature"].values)
    plt.show()
    #
    # # Find Temperature-Irradiation Dependency
    # sim = Simulator()
    # temp = Temperature()
    # time, irradiation = sim.simulate_sun()
    #
    # data = pd.DataFrame("Generator-TestData-2021-08-30--2022-08-30.csv")
    #
    # plt.plot(data["time"], data[""])
    #
    #


    # t = temp.stefan_boltzmann(irradiation)+10
    # print(t)
    # plt.plot(time, t)
    # plt.show()


