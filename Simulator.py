from .Panel_Simulation import Panel_Simulator
from .Irradiation import Sun_Simulator
from .Shading import Shading
from .DataSheet_Manager import DataSheet_Manager
from .Temperature_Simulation import Temperature
import numpy as np
import pandas as pd
import os
import time as time_bib_sleep
from tqdm import tqdm


class Simulator(DataSheet_Manager, Sun_Simulator, Panel_Simulator, Shading, Temperature):
    def __init__(self,start_date="2022-07-24", end_date="2022-07-25",frequence = 10, panel_name = "erlangen_new", module_id = "Test123"):
        """
        :param start_date:
        :param end_date:
        :param frequence:
        :param panel_name:
        :param module_type:
        :param n_stings_per_modul:
        :param n_waver_per_string:
        :param module_ISC: short circuit current in W
        :param module_shunt_resistance: parallel resistor in OHM
        :param module_reverse_leakage_current: in nA
        :param module_serial_resistance: serial resistance in OHM
        """

        DataSheet_Manager.__init__(self, panel_name, module_id)

        Sun_Simulator.__init__(self, start_date=start_date, end_date=end_date, frequence=frequence,
                               longitude=self.origin_longitude, latitude=self.origin_latitude,
                               panel_altitude=self.origin_panel_altitude, tilt_angle=self.origin_tilt_angle,
                               azimuth_angle=self.origin_azimuth_angle, albedo = self.origin_albedo)

        Panel_Simulator.__init__(self, module_type=self.origin_module_type)
        Shading.__init__(self, latitude=self.latitude, longitude=self.longitude,
                         start_date=start_date, end_date=end_date, frequence=frequence)
        Temperature.__init__(self, self.module_width, self.module_length)
    
        self.UI_curve = None
        self.simulation = pd.DataFrame()
        self.map_size_x, self.map_size_y = None,None

    ################################ -- Sun Simulator -- #######################################################
    def get_latitude(self):
        return self.latitude
    def set_latitude(self, latitude):
        self.latitude = latitude

    def get_longitude(self):
        return self.longitude
    def set_longitude(self, longitutde):
        self.longitude = longitutde

    def get_azimuth(self):
        if self.azimuth is None: self.tasa_get_azimuth()
        return self.azimuth
    def set_azimuth(self, azimuth):
        self.azimuth = azimuth
    def get_altitude(self):
        if self.altitude is None: self.tasa_get_altitude()
        return self.altitude
    def set_altitude(self, altitude):
        return self.altitude
    def get_time(self):
        self.get_dates_pandas()
        return self.dates_pandas
    def set_time(self, time):
        self.data_time = time
        
    def get_tilt_angle(self):
        return self.tilt_anlge
    def set_tilt_angle(self,tilt_angle):
        self.tilt_anlge = tilt_angle
        
    def get_azimuth_angle(self):
        return self.azimuth_angle
    def set_azimuth_angle(self,azimuth_angle):
        self.azimuth_angle = azimuth_angle
        
    def get_albedo(self):
        return self.albedo
    def set_albedo(self,albedo):
        self.albedo = albedo
    
    def get_panel_altitude(self):
        return self.panel_altitude
    def set_panel_altitude(self,panel_altitude):
        self.panel_altitude = panel_altitude
        
        
    def randomize_simulation_irradiance_on_plane(self, panel_altitude=(-50,50), tilt_angle=(-10,10),
                                                 azimuth_angle=(-10,10), albedo=(-0.1,0.1)):
        albedo = np.array(albedo)*100
        self.panel_altitude = self.origin_panel_altitude + np.random.randint(panel_altitude[0],panel_altitude[1])
        self.tilt_angle = self.origin_tilt_angle + np.random.randint(tilt_angle[0],tilt_angle[1])
        self.azimuth_angle = self.origin_azimuth_angle + np.random.randint(azimuth_angle[0],azimuth_angle[1])
        self.albedo = self.origin_albedo + np.random.randint(albedo[0],albedo[1])/100
    def get_irradiance_on_plane(self):
        """
        Simulates sun. Uses "Irradiation import Sun_Simulator.irradiance_on_plane" to calculate irradiance on plane.
        :return: time, irradiation
        """
        return self.irradiance_on_plane()
    def set_irradiance_on_plane(self, irradiance):
        self.irradiance = irradiance

    def get_sun_simulation(self):
        data = pd.DataFrame()
        data["time"] = self.get_time()
        data["azimuth"] = self.get_azimuth()
        data["altitude"] = self.get_altitude()
        data["irradiance"] = self.get_irradiance_on_plane()
        return data
    #############################################################################################################

    def get_module_pos_list(self,shading_object_max_range, module_rows, module_columns, module_space_rows=0, module_space_columns=0,
                            string_rows=1, strings_columns=1, string_space_rows=0, string_space_columns=0):
        return shading_set_modules(shading_object_max_range, module_rows, module_columns, module_space_rows, module_space_columns,
                            string_rows, strings_columns, string_space_rows, string_space_columns)
    def set_module_pos_list(self, module_pos_list):
        self.module_pos_list = module_pos_list

    ##############################################################################################################
    ##############--------- IRRADIATION CHANGES -----------------#################################################
    def map_define_size(self, cubes=[[[40, 40, 10], [10, 20, 100], 0.1],
                                     [[400,400,0],[20,50,50],0.2]], module = [100,100]):
        """
        Takes the edge poitns of the cubes and modules, to create a map with minimal size.
        This function is used within "simulate_and_add_shading_to_waver_list.
        :return: map_x-size, map_y-size
        """
        if cubes is not None:
            xs = [cube[0][0] + cube[1][0] for cube in cubes]
            ys = [cube[0][1] + cube[1][1] for cube in cubes]
        else:
            xs=[]
            ys=[]
        if len(np.array(module).shape)<2:
            xs.append(module[0]+self.module_width)
            ys.append(module[1]+self.module_length)
        else:
            module = module.reshape(-1, 2)
            xm = module[:,0]+self.module_width
            ym = module[:,1]+self.module_length
            xs = np.append(xs,xm)
            ys = np.append(ys,ym)
            #print(xs,ys)
        self.map_size_x, self.map_size_y = np.max(xs), np.max(ys)
        #print(f"MAP BUILD: {self.map_size_x}|{self.map_size_y}")
        return self.map_size_x, self.map_size_y
    def map_refresh(self):
        self.build_map(self.map_size_x, self.map_size_y)  # CREATE MAP
    def map_get_module_irradiation_factor_matrix(self, module = [100, 100], waver_shape=(10, 20)):
        m = [[module[0], module[1]],[self.module_width, self.module_length]]
        matrix = self.get_module_irradiation(m)
        return self.calc_waver_mean(matrix, waver_shape) # calculates the irradiation for each single waver
    def get_module_edges_for_plot(self, module):
        x, y, w, l = module[0], module[1], self.module_width, self.module_length
        x1, x2 = x, w + x
        y1, y2 = y, l + y
        return [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]

    ###############-------- SHADING-----------####################################################################
    def map_update_shading(self, frame, cubes):
        """Add Shadow to the MAP
        retuns: Shadow map"""
        return self.add_cubes_to_map(frame=frame, cubes=cubes)

    def get_shaded_module_list(self, shadow_map, module_pos_list = [[100,100], [100,200]]):
        """Checks if a module is shaded. If so its shadow-value gets 1.
        Returns a list of 0 and 1 for each module."""
        shadow_list = np.zeros(len(module_pos_list))
        for n,m in enumerate(module_pos_list):
            shading = np.sum(shadow_map[m[1]:m[1]+self.module_length,m[0]:m[0]+self.module_width])
            if shading > 0: shadow_list[n] = 1
            else: None
        return shadow_list

    def define_random_cube_for_shading(self, height_range = [0,300], length_range = [10,200], width_range=[10,200], module_pos_list = np.array([[100,100], [100,200]])):
        h = np.random.randint(height_range[0], height_range[1])
        b, w = np.random.randint(length_range[0],length_range[1]), np.random.randint(width_range[0], width_range[1])
        fh = 0
        y_space = 410
        x_space = 410
        width = x_space * 2 + 120 * 8
        length = y_space * 2 + 70 * 2

        day_frames = len(self.azimuth)
        f1,f2,f3 = int(day_frames/4),int(day_frames/2), int(day_frames*0.75)
        azis = self.azimuth[[f1,f2,f3,]]
        #print(np.array(azis/np.pi*2, dtype=np.int32))
        sl = h / np.tan(self.altitude[[f1,f2,f3]])
        rander = np.random.randint(0, len(azis))
        azimuth = azis[rander]
        shadow_length = sl[rander]
        xx,yy = np.sin(azimuth) * shadow_length, np.cos(azimuth) * shadow_length
        x,y = abs(xx),abs(yy)
        #print(f"X: {xx}; Y: {yy}   [Simulator-define_random_cube_for_shading]")

        side = int(azimuth/np.pi*2)
        #print(f"SIDE: {side}   [Simulator-define_random_cube_for_shading]")
        minx = np.min(module_pos_list[:,:, 0])
        maxx = np.max(module_pos_list[:,:, 0])+self.module_width
        miny = np.min(module_pos_list[:,:,1])
        maxy = np.max(module_pos_list[:,:, 1])+ self.module_length

        if miny-y-b <= 0: top = 0
        else: top = miny-y-b

        if maxy+y <= length-b: bot = maxy+y
        else: bot = length-b

        if minx-x-w <= 0: left = 0
        else: left = minx-x-w

        if maxx+x >= width-w: right = width-w
        else: right = width-w

        if (side == 0 | side == 4):  # top
            #print(left, right, top, miny - b)
            y_pos = np.random.randint(top, miny - b)
            x_pos = np.random.randint(left, right)
        elif side == 1:  # left
            #print(left, minx-w ,top, bot)
            #if yy <=0:
            y_pos = np.random.randint(miny-b//2, bot-b)
            #else: y_pos = np.random.randint(top, maxy)
            x_pos = np.random.randint(left, minx-w)
        elif side == 2:  # bot
            #print(left, right, maxy, bot)
            y_pos = np.random.randint(maxy, bot)
            x_pos = np.random.randint(left, right)
        else:  # right
            #print(maxx,right, top, bot)
            #if yy <= 0:
            y_pos = np.random.randint(miny - b//2, bot-b)
            #else:y_pos = np.random.randint(top, maxy)
            x_pos = np.random.randint(maxx,right)
        cube = [[x_pos,y_pos,fh], [w,b,h],0.4]
        return cube

    ###############-------- Clouds -------------####################################################################
    def map_update_noiseclouds(self, couldy_factor=1, variation=0):
        self.add_noise_cloudy_sky(couldy_factor, variation)

    def simulate_module(self, cubes=None, module = [0,0]):
        """
        Simulates voltage, current, and power of a module over the start-end time.
        Therefore the Simulation always gives the MPP information, since a perfect MPP-Tracker is assumed.
        Shading can be included
        :param shading: if True shading is included
        :param cubes: only important when shading is true. Adds cubes which produce shading
        :param module: only important when shading is true. Sets position of module.
        :return: pandas data frame, with time, voltage, current and power
        """
        self.map_define_size(cubes, module)
        time = self.get_time()
        irradiation = self.get_irradiance_on_plane()
        U_list, I_list, P_list = [],[],[]
        for f,irr in enumerate(tqdm(irradiation)):
            waver_list = self.create_module_matrix(irr)
            if cubes is not None: waver_list[:,:,0]*=self.map_get_module_irradiation_factor_matrix(f, cubes, module, waver_list.shape)
            U_Mpp, I_Mpp, P_Mpp = self.get_MPP_perLTSpice(waver_list)
            U_list.append(U_Mpp)
            I_list.append(I_Mpp)
            P_list.append(P_Mpp)
        self.simulation["time"] = time
        self.simulation["voltage"] = U_list
        self.simulation["current"] = I_list
        self.simulation["power"] = P_list
        return self.simulation



