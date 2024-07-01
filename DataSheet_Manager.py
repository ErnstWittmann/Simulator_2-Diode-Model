import os
import pandas as pd

class DataSheet_Manager():
    def __init__(self, panel_name, module_id):
        self.__initialize__(panel_name, module_id)
    ######### - File organisation - ####################################################################################
    def __search_for_location_csv__(self):
        """
        Searches for "Locations.csv". This csv-Frame includes the location-information like longitude, latidude ...
        If no Location.csv exists, a csv with the location information of "erlangen" will be created.
        """
        name = "Locations.csv"
        if not os.path.exists(name):
            print("NO Locations-File FOUND!!!")
            loc = pd.DataFrame(data={"Location": ["erlangen_new"]})
            loc["Latitude"] = 49.583332
            loc["Longitude"] = 11.016667
            loc["Altitude"] = 300
            loc["Azimuth_Angle"] = 135
            loc["Tilt_Angle"] = 20
            loc["Albedo"] = 0.23
            print(loc)
            loc.to_csv("Locations.csv")
            print("Location-File was Created!!")
        else:
            print("Locations-File found !")


    def __read_location_csv__(self, panel_name):
        params = pd.read_csv("Locations.csv", delimiter=",")
        params = params.loc[params["Location"] == panel_name]
        self.origin_longitude = float(params["Longitude"].values[0])
        self.origin_latitude = float(params["Latitude"].values[0])
        self.origin_panel_altitude = int(params["Altitude"].values[0])
        self.origin_tilt_angle = float(params["Tilt_Angle"].values[0])
        self.origin_azimuth_angle = float(params["Azimuth_Angle"].values[0])
        self.origin_albedo = float(params["Albedo"].values[0])
        print("Location-File readed!!")


    def __search_for_LTSpice_Module_Parameter_csv__(self):
        """
        Searches for "LTSpice_Module_Parameter.csv". This csv-Frame includes the Module-information like module_type, length, width ...
        If no LTSpice_Module_Parameter.csv exists, a csv with the Test-Module-Information will be created.
        To find good module parameter measuere the UI-Curve of a module and than use the MonteCarloTreeSearch to find the Module parameter.
        """
        name = "LTSpice_Module_Parameter.csv"
        if not os.path.exists(name):
            print("NO LTSpice_Module_Parameter-File FOUND!!!")
            loc = pd.DataFrame(data={"Module_ID": ["Test123"]})
            loc["module_type"] = "full-cell_module"
            loc["module_length in cm"] = 30
            loc["module_width in cm"] = 60
            loc["n_stings_per_modul"] = 3
            loc["n_waver_per_string"] = 20
            loc["ISC in A"] = 5.17
            loc["shunt_resistance in Ohm"] = 100
            loc["good_diode_reverse_leakage_current in nA"] = 1e-9
            loc["bad_diode_reverse_leakage_current in nA"] = 1e-9
            loc["good_diode_N"] = 2
            loc["bad_diode_N"] = 1
            loc["serial_resistance in Ohm"] = 0.1
            loc.to_csv(name, sep=';')
            print("LTSpice_Module_Parameter-File was Created!!")
        else:
            print("LTSpice_Module_Parameter-File found !")


    def __read_LTSpice_Module_Parameter_csv__(self, module_id):
        params = pd.read_csv("LTSpice_Module_Parameter.csv", delimiter=";")
        params = params.loc[params["Module_ID"] == module_id]
        self.n_strings_per_modul = int(params["n_stings_per_modul"].values[0])
        self.n_waver_per_string = int(params["n_waver_per_string"].values[0])
        self.module_length = int(params["module_length in cm"].values[0])
        self.module_width = int(params["module_width in cm"].values[0])
        self.origin_module_type = params["module_type"].values[0]
        self.origin_module_ISC = float(params["ISC in A"].values[0])
        self.origin_module_shunt_resistance = float(params["shunt_resistance in Ohm"].values[0])
        self.origin_module_good_diode_reverse_leakage_current = float(
            params["good_diode_reverse_leakage_current in nA"].values[0])
        self.origin_module_bad_diode_reverse_leakage_current = float(
            params["bad_diode_reverse_leakage_current in nA"].values[0])
        self.origin_module_good_diode_N = float(params["good_diode_N"].values[0])
        self.origin_module_bad_diode_N = float(params["bad_diode_N"].values[0])
        self.origin_module_serial_resistance = float(params["serial_resistance in Ohm"].values[0])
        self.module_type = params["module_type"].values[0]
        self.module_ISC = float(params["ISC in A"].values[0])
        self.module_shunt_resistance = float(params["shunt_resistance in Ohm"].values[0])
        self.module_good_diode_reverse_leakage_current = float(params["good_diode_reverse_leakage_current in nA"].values[0])
        self.module_bad_diode_reverse_leakage_current = float(params["bad_diode_reverse_leakage_current in nA"].values[0])
        self.module_good_diode_N = float(params["good_diode_N"].values[0])
        self.module_bad_diode_N = float(params["bad_diode_N"].values[0])
        self.module_serial_resistance = float(params["serial_resistance in Ohm"].values[0])
        print("Module Parameter readed")


    def add_LTSpice_Module_Parameter_to_csv(self, module_Id, module_type,
                                            module_lenght, module_width, strings_per_module, waver_per_string,
                                            module_params):
        loc = pd.DataFrame(data={"Module_ID": [module_Id]})
        loc["module_type"] = module_type
        loc["module_length in cm"] = module_lenght
        loc["module_width in cm"] = module_width
        loc["n_stings_per_modul"] = strings_per_module
        loc["n_waver_per_string"] = waver_per_string
        loc["ISC in A"] = module_params[0]
        loc["shunt_resistance in Ohm"] = module_params[1]
        loc["good_diode_reverse_leakage_current in nA"] = module_params[2]
        loc["bad_diode_reverse_leakage_current in nA"] = module_params[3]
        loc["good_diode_N"] = module_params[4]
        loc["bad_diode_N"] = module_params[5]
        loc["serial_resistance in Ohm"] = module_params[6]
        params = pd.read_csv("Locations.csv", delimiter=",")
        params = params.append(loc)
        params.to_csv("LTSpice_Module_Parameter.csv", sep=';')


    def __initialize__(self, panel_name, module_id):
        # DEFINE SUN PARAMETER
        self.__search_for_location_csv__()
        self.__read_location_csv__(panel_name)

        # DEFINE MODULE PARAMETER
        self.__search_for_LTSpice_Module_Parameter_csv__()
        self.__read_LTSpice_Module_Parameter_csv__(module_id)