from .TimeAndSunAngle import TimeAndSunAngle

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pysolar import solar, radiation
from tqdm import tqdm



class Shading(TimeAndSunAngle):
    # with this programm shading can be added to the Simulator.
    def __init__(self, latitude=11, longitude=40, start_date="2022-07-02", end_date="2022-07-03", frequence=10):
        """
        Used to calculate shading done by objects or clouds.

        First a Map has to be created by build_map(x_size, y_size). Therefore you need to enter the size in cm.
        The Map will include the irradiance-factor. The Map has an resolution of 1cm^2 per pixel.

        Afterwards with add_cubes_to_map(frame, cubes) you can add objects to the map which will produce shaddow.
        There are also some prepared cube-combinations avaiable.
        With add_random_cloud(cloud_factor, scatter_factor) also clouding can be added to the map.

        You can see the module-radiation via get_module_irradiation(module) or plot the modules via plot_module(module,ax)

        In the end use calc_waver_mean to get the irradiation-factor for your modules. The return of this function gives you
        an array of factors which can be used to factor your irradiation of the modules. The result can afterwards be used
        by the Simulator.py to simulate the output of your modules.



        :param latitude: latitude of place (use google to find it)
        :param longitude: longitude of place (use google to find it)
        :param start_date: the date in YYYY-MM-DD when the simulation starts
        :param end_date: the date in YYYY-MM-DD when the simulation ends
        :param frequence: frequency in MIN -> each ...Min a frame is calculated
        """
        TimeAndSunAngle.__init__(self, start_date=start_date, end_date=end_date, frequence=frequence,
                                 longitude=longitude, latitude=latitude)
        self.cloud_pos = np.array([0,0])

    def build_map(self, x_size, y_size):
        """
        builds a map where cubes and modules can be placed on.
        Map will include the irradiation factor.
        Multiply the map with an irradiation and it will give you the irradiation in a resolution of 1cm^2.
        :param x_size: x size of map in cm
        :param y_size: y size of map in cm
        """
        # If azimuth and altitude are not calculated yet, than do so
        # Azimuth is the angle of West-Ost, altitude is the angle of surface to sun
        # we need both angles to calculate the shadow produced by sun
        if self.azimuth is None: self.get_azimuth()
        if self.altitude is None: self.get_altitude()

        # create mask-Map
        self.map = np.ones((y_size,x_size)) # 1 pixel =1 cm^2

        # create coordinate xy-map
        x_map = np.ones_like(self.map) * np.arange(x_size)[None,:]
        y_map = np.ones_like(self.map) * np.arange(y_size)[:,None]
        self.xy_map = np.ones((y_size,x_size,2))
        self.xy_map[:,:,0] = x_map
        self.xy_map[:,:,1] = y_map
        #print(self.xy_map.shape)

    def get_module_irradiation(self, module = [[100,100], [40,60]]):#, rotation=45):
        """
        takes the module irradiation values out of the map.
        returns a np.array including the irradiations for each cm^2
        module is similar to cubes:
        module = [[y-pos, x-pos], [length, width]]"""
        x,y = module[0][0], module[0][1]
        w,l = module[1][0], module[1][1]
        return self.map[y:y+l, x:x+w]

    def calc_waver_mean(self, module_irradiation_matrix, waver_shape, module_structure="U-norm"):
        """used by Simulator.py
        takes the irraditaion of the modules and compromises it, so it can be used by the simulator.
        """
        # calculates for each waver the shading
        #print(module_irradiation_matrix.shape)
        #print(waver_shape)
        mim_0 = module_irradiation_matrix.shape[0]
        mim_1 = module_irradiation_matrix.shape[1]

        if module_structure == "Z-norm":
            # the module structure has form
            #  w1-w2-w3-.
            # .---------'
            # '-w4-w5-w6
            # but the since waver list has no knick in string, the wavermatrix looks like
            # w1-w2-w3-w4-w5-w6
            ws_0 = int(waver_shape[0]*2) # since each string of module takes 2 rows
            ws_1 = int(waver_shape[1]/2) # since each string of module is kicked by half lenght
        if module_structure == "U-norm":
            # the module structure has form
            #  w1-w2-w3-.
            #  w6-w5-w4-'
            # but the since waver list has no knick in string, the wavermatrix looks like
            # w1-w2-w3-w4-w5-w6
            ws_0 = int(waver_shape[0] * 2)  # since each string of module takes 2 rows
            ws_1 = int(waver_shape[1] / 2)  # since each string of module is kicked by half lenght

        a = module_irradiation_matrix.reshape(mim_0, ws_1,-1)
        b = np.sum(a, axis=-1)
        c = b.T
        c = c.reshape(ws_1, ws_0, -1)
        d = np.sum(c, axis=-1)/(mim_0/ws_0*mim_1/ws_1) # mean value
        e = d.T


        if module_structure == "Z-norm": e = e.reshape(int(ws_0/2),int(ws_1*2))
        if module_structure == "U-norm":
            i = e[::2]
            u = e[1::2,::-1]
            e = np.append(i,u, axis=1)
        return e

    def plot_module(self, module, ax=None):#, rotation=45):
        """returns the corner points of one module."""
        x, y, w, l = module[0][0], module[0][1], module[1][0], module[1][1]
        x1,x2 = x, w+x
        y1,y2 = y, l+y
        if ax==None:plt.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], "red")
        else: ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], "red")

    def get_map(self):
        return self.map


    def shading_get_modules(self, shading_object_max_range, module_rows, module_columns, module_space_rows=0, module_space_columns=0,
                            string_rows=1, strings_columns=1, string_space_rows=0, string_space_columns=0):
        string_list = []
        for sr in range(string_rows):
            for sc in range(strings_columns):
                module_list = []
                for mr in range(module_rows):
                    for mc in range(module_columns):
                        module_list.append([shading_object_max_range+self.module_width*mr+module_space_rows*mr+
                                            (self.module_width*module_rows+module_space_rows*(module_rows-1)+string_space_rows)*sr,
                                            shading_object_max_range+self.module_length*mc + module_space_columns*mc+
                                            (self.module_length*module_columns+module_space_columns*(module_columns-1)+string_space_columns)*sc])
                string_list.append(module_list)
        self.module_pos_list = np.array(string_list)

        return self.module_pos_list

    ######################## ---- SHADOW ----# ########################################################
    def add_cubes_to_map(self, frame, cubes=[[[40,40,10],[10,20,100],0.1],[[100,400,0],[20,50,50],0.2]]):
        # [x-pos, y-pos][width, length, height in [cm]], irradiance_reduction
        """
        first you need to create a map with build_map(x_size, y_size).
        takes a frame and calculates the shadow of the cubes.
        generates the shadow and cubes on the map. Map is a 2D-array including all irradiantions. Each pixel of the map is 1cm^2
        :param frame: frame, the timestep where altitude and longitude have been calculated already
        :param cubes: [x-pos, y-pos][width, length, height in [cm]], irradiance_reduction
        :return: an inverted map


        "cubes contains a list of single cubes:"
              "each cube of cubes consists two lists and one value:"
              "the first list consists the x, y and z position of the left,top,bottom point of the cube"
              "the second the width, lenght and height of the cube"
              "everything is in cm"

              "General:"
              "cube = [[x_pos,y_pos, z_pos],[width,lenght,height], irradiation_reduction_factor]"
              "cubes = [cube1,cube2,cube3]"
        """
        # tmap = shadow map
        tmap = np.zeros_like(self.map)
        if self.altitude[frame]>=0:
            #print("calculate")
            for cube in cubes:
                # calculate cube top points
                cube_x = cube[0][0], cube[0][0]+cube[1][0] #x pos of object
                cube_y = cube[0][1], cube[0][1]+cube[1][1] #y pos of object

                # calculate shading
                shadow_lenght = (cube[1][2]+cube[0][2])/np.tan(self.altitude[frame])                        # length of shadow of object in cm
                no_shadow_lenght = cube[0][2]/np.tan(self.altitude[frame])                                  # if the object is flying

                shadow_area_y = np.array(cube_y + np.cos(self.azimuth[frame])*shadow_lenght).astype(np.int32) #x pos of shadow projection in cm
                shadow_area_x = np.array(cube_x + np.sin(self.azimuth[frame])*shadow_lenght).astype(np.int32) #y pos of shadow projection in cm
                no_shadow_area_y = np.array(cube_y + np.cos(self.azimuth[frame])*no_shadow_lenght).astype(np.int32) #x pos of shadow projection in cm
                no_shadow_area_x = np.array(cube_x + np.sin(self.azimuth[frame]) * no_shadow_lenght).astype(np.int32) #y pos of shadow projection in cm
                # print("cube_x: ", cube_x)
                # print("cube_y: ", cube_y)
                # print("shadow_lenght: ",shadow_lenght)
                # print("no_shadow_lenght: ", no_shadow_lenght)
                # print("shadow_area_y: ", shadow_area_y)
                # print("shadow_area_x: ", shadow_area_x)
                # print("no_shadow_area_y: ", no_shadow_area_y)
                # print("no_shadow_area_x: ", no_shadow_area_x)
                # print("azimuth and cos: ",self.azimuth[frame], np.cos(self.azimuth[frame]))

                # where object is placed, there is no light
                self.map[cube_y[0]:cube_y[1], cube_x[0]:cube_x[1]] = 0
                # Calculate shadow
                m = (cube_y[0] - shadow_area_y[0]) / (cube_x[0] - shadow_area_x[0]+ np.finfo(np.float32).eps)
                if (shadow_area_y[0]< cube_y[0]) & (shadow_area_x[0]>=cube_x[0]):
                    #print("case 1")
                    #    .....
                    #   /    /
                    #  /    /
                    # /----/
                    t_up = cube_y[0]-m*cube_x[0]
                    t_down = cube_y[1]-m*cube_x[1]

                    if cube_x[0] - shadow_area_x[0] == 0:
                        tmap[(self.xy_map[:, :, 0] > no_shadow_area_x[0]) &
                                 (self.xy_map[:, :, 0] < shadow_area_x[1]) &
                                 (self.xy_map[:, :, 1] < no_shadow_area_y[1]) &
                                 (self.xy_map[:, :, 1] > shadow_area_y[0])]=cube[2]
                    else:
                        tmap[(self.xy_map[:,:,0]>no_shadow_area_x[0]) &
                            (self.xy_map[:,:,0]<shadow_area_x[1])&
                            (self.xy_map[:,:,1]<no_shadow_area_y[1])&
                            (self.xy_map[:,:,1]>shadow_area_y[0])&
                            (self.xy_map[:,:,0]*m+ t_up < self.xy_map[:,:,1])&
                            (self.xy_map[:,:,0]*m+ t_down > self.xy_map[:,:,1])]=cube[2]
                elif (shadow_area_x[0]>=cube_x[0]) & (shadow_area_y[0]>= cube_y[0]):
                    #print("case 2")
                    #    -----
                    #   /    /
                    #  /    /
                    # /..../
                    t_up = cube_y[0] - m * cube_x[1]
                    t_down = cube_y[1] - m * cube_x[0]
                    if cube_x[0] - shadow_area_x[0] == 0:
                        tmap[(self.xy_map[:, :, 0] >= no_shadow_area_x[0]) &
                                 (self.xy_map[:, :, 0] <= shadow_area_x[1]) &
                                 (self.xy_map[:, :, 1] < shadow_area_y[1]) &
                                 (self.xy_map[:, :, 1] > no_shadow_area_y[0])]=cube[2]
                    else:
                        tmap[(self.xy_map[:, :, 0] >= no_shadow_area_x[0]) &
                            (self.xy_map[:, :, 0] <= shadow_area_x[1]) &
                            (self.xy_map[:, :, 1] < shadow_area_y[1]) &
                            (self.xy_map[:, :, 1] > no_shadow_area_y[0]) &
                            (self.xy_map[:, :, 0] * m + t_up < self.xy_map[:, :, 1]) &
                            (self.xy_map[:, :, 0] * m + t_down > self.xy_map[:, :, 1])]=cube[2]
                elif (shadow_area_x[0]<cube_x[0]) & (shadow_area_y[0]>=cube_y[0]):
                    #print("case 3")
                    t_up = cube_y[0] - m * cube_x[0]
                    t_down = cube_y[1] - m * cube_x[1]
                    if cube_x[0] - shadow_area_x[0] == 0:
                        tmap[(self.xy_map[:, :, 0] > shadow_area_x[0]) &
                                 (self.xy_map[:, :, 0] < no_shadow_area_x[1]) &
                                 (self.xy_map[:, :, 1] < shadow_area_y[1]) &
                                 (self.xy_map[:, :, 1] > no_shadow_area_y[0])]=cube[2]
                    else:
                        tmap[(self.xy_map[:, :, 0] > shadow_area_x[0]) &
                            (self.xy_map[:, :, 0] < no_shadow_area_x[1]) &
                            (self.xy_map[:, :, 1] < shadow_area_y[1]) &
                            (self.xy_map[:, :, 1] > no_shadow_area_y[0]) &
                            (self.xy_map[:, :, 0] * m + t_up < self.xy_map[:, :, 1]) &
                            (self.xy_map[:, :, 0] * m + t_down > self.xy_map[:, :, 1])]=cube[2]
                elif (shadow_area_x[0]<cube_x[0]) & (shadow_area_y[0]<cube_y[0]):
                    #print("case 4")
                    t_up = cube_y[0] - m * cube_x[1]
                    t_down = cube_y[1] - m * cube_x[0]
                    if cube_x[0] - shadow_area_x[0] == 0:
                        tmap[(self.xy_map[:, :, 0] > shadow_area_x[0]) &
                                 (self.xy_map[:, :, 0] < no_shadow_area_x[1]) &
                                 (self.xy_map[:, :, 1] < no_shadow_area_y[1]) &
                                 (self.xy_map[:, :, 1] > shadow_area_y[0])]=cube[2]
                    else:
                        #print("x:",shadow_area_x[0],no_shadow_area_x[1])
                        #print("y:", shadow_area_y[0], no_shadow_area_y[1])
                        tmap[(self.xy_map[:, :, 0] >= shadow_area_x[0]) &
                                (self.xy_map[:, :, 0] <= no_shadow_area_x[1]) &
                                (self.xy_map[:, :, 1] < no_shadow_area_y[1]) &
                                (self.xy_map[:, :, 1] > shadow_area_y[0]) &
                                (self.xy_map[:, :, 0] * m + t_up < self.xy_map[:, :, 1]) &
                                (self.xy_map[:, :, 0] * m + t_down > self.xy_map[:, :, 1])]=cube[2]
                else: print("HÄ?")
        else:
            #print("Night")
            self.map = np.zeros_like(self.map)
        self.map -= tmap
        self.map[self.map<0]=0
        return tmap

    def add_pyramid_to_map(self, frame, pyramids=[[[40,40,10],[10,20,100],0.1],[[100,400,0],[20,50,50],0.2]]):
        if self.altitude[frame]>=0:
            tmap = np.zeros_like(self.map)
            for pyr in pyramids:
                # calculate cube top points
                pyr_x = pyr[0][0],pyr[0][0]+pyr[0][1]
                pyr_y = pyr[0][1],pyr[0][1]+pyr[1][1]
                top_x = pyr[0][0]+np.array(pyr[1][0])/2
                top_y = pyr[0][1]+np.array(pyr[1][1])/2
                # calculate shading
                shadow_lenght = (pyr[1][2] + pyr[0][2]) / np.tan(self.altitude[frame])  # length of shadow of object in cm
                no_shadow_lenght = pyr[0][2] / np.tan(self.altitude[frame])
                shadow_area_y = np.array(top_y + np.cos(self.azimuth[frame]) * shadow_lenght).astype(np.int32)  # x pos of shadow projection in cm
                shadow_area_x = np.array(top_x + np.sin(self.azimuth[frame]) * shadow_lenght).astype(np.int32)  # y pos of shadow projection in cm
                no_shadow_area_y = np.array(top_y + np.cos(self.azimuth[frame]) * no_shadow_lenght).astype(np.int32)  # x pos of shadow projection in cm
                no_shadow_area_x = np.array(top_x + np.sin(self.azimuth[frame]) * no_shadow_lenght).astype(np.int32)  # y pos of shadow projection in cm

                self.map[pyr[0][1]:pyr[0][1]+pyr[1][1], pyr[0][0]:pyr[0][0]+pyr[0][1]] = 0
                # Calculate shadow

                if (shadow_area_y < pyr_y[0]) & (shadow_area_x >= pyr_x[0]):
                    #\''\
                    #  \\
                    #   '
                    print("case 1")
                    mup = (pyr_y[0] - shadow_area_y) / (pyr_x[0] - shadow_area_x + np.finfo(np.float32).eps)
                    mdw = (pyr_y[1] - shadow_area_y) / (pyr_x[1] - shadow_area_x + np.finfo(np.float32).eps)
                    t_up = pyr_y[0] - mup * pyr_x[0]
                    t_down = pyr_y[1] - mdw * pyr_x[1]

                    if pyr_x[0] - shadow_area_x == 0:
                        tmap[(self.xy_map[:, :, 0] > no_shadow_area_x[0]) &
                             (self.xy_map[:, :, 0] < shadow_area_x[1]) &
                             (self.xy_map[:, :, 1] < no_shadow_area_y[1]) &
                             (self.xy_map[:, :, 1] > shadow_area_y[0])] = pyr[2]
                    else:
                        tmap[(self.xy_map[:, :, 0] > no_shadow_area_x[0]) &
                             (self.xy_map[:, :, 0] < shadow_area_x[1]) &
                             (self.xy_map[:, :, 1] < no_shadow_area_y[1]) &
                             (self.xy_map[:, :, 1] > shadow_area_y[0]) &
                             (self.xy_map[:, :, 0] * mup + t_up < self.xy_map[:, :, 1]) &
                             (self.xy_map[:, :, 0] * mdw + t_down > self.xy_map[:, :, 1])] = pyr[2]
                elif (shadow_area_x >= pyr_x[0]) & (shadow_area_y >= pyr_y[0]):
                    print("case 2")
                    mup = (pyr_y[0] - shadow_area_y) / (pyr_x[1] - shadow_area_x + np.finfo(np.float32).eps)
                    mdw = (pyr_y[1] - shadow_area_y) / (pyr_x[0] - shadow_area_x + np.finfo(np.float32).eps)
                    t_up = pyr_y[0] - mup * pyr_x[1]
                    t_down = pyr_y[1] - mdw * pyr_x[0]
                    if pyr_x[0] - shadow_area_x == 0:
                        tmap[(self.xy_map[:, :, 0] >= no_shadow_area_x[0]) &
                             (self.xy_map[:, :, 0] <= shadow_area_x[1]) &
                             (self.xy_map[:, :, 1] < shadow_area_y[1]) &
                             (self.xy_map[:, :, 1] > no_shadow_area_y[0])] = pyr[2]
                    else:
                        tmap[(self.xy_map[:, :, 0] >= no_shadow_area_x) &
                             (self.xy_map[:, :, 0] <= shadow_area_x) &
                             (self.xy_map[:, :, 1] < shadow_area_y) &
                             (self.xy_map[:, :, 1] > no_shadow_area_y) #&
                             #(self.xy_map[:, :, 0] * mup + t_up < self.xy_map[:, :, 1]) &
                             #(self.xy_map[:, :, 0] * mdw + t_down > self.xy_map[:, :, 1])
                             ] = pyr[2]
                elif (shadow_area_x < pyr_x[0]) & (shadow_area_y >= pyr_y[0]):
                    print("case 3")
                    mup = (pyr_y[0] - shadow_area_y) / (pyr_x[0] - shadow_area_x + np.finfo(np.float32).eps)
                    mdw = (pyr_y[1] - shadow_area_y) / (pyr_x[1] - shadow_area_x + np.finfo(np.float32).eps)
                    t_up = pyr_y[0] - mup * pyr_x[0]
                    t_down = pyr_y[1] - mdw * pyr_x[1]
                    if pyr_x[0] - shadow_area_x == 0:
                        tmap[(self.xy_map[:, :, 0] > shadow_area_x) &
                             (self.xy_map[:, :, 0] < no_shadow_area_x) &
                             (self.xy_map[:, :, 1] < shadow_area_y) &
                             (self.xy_map[:, :, 1] > no_shadow_area_y)] = pyr[2]
                    else:
                        tmap[(self.xy_map[:, :, 0] > shadow_area_x) &
                             (self.xy_map[:, :, 0] < no_shadow_area_x) &
                             (self.xy_map[:, :, 1] < shadow_area_y) &
                             (self.xy_map[:, :, 1] > no_shadow_area_y) &
                             (self.xy_map[:, :, 0] * mup + t_up < self.xy_map[:, :, 1]) &
                             (self.xy_map[:, :, 0] * mdw + t_down > self.xy_map[:, :, 1])] = pyr[2]
                elif (shadow_area_x < pyr_x[0]) & (shadow_area_y < pyr_y[0]):
                    mup = (pyr_y[0] - shadow_area_y) / (pyr_x[1] - shadow_area_x + np.finfo(np.float32).eps)
                    mdw = (pyr_y[1] - shadow_area_y) / (pyr_x[0] - shadow_area_x + np.finfo(np.float32).eps)
                    print("case 4")
                    t_up = pyr_y[0] - mup * pyr_x[1]
                    t_down = pyr_y[1] - mdw * pyr_x[0]
                    if pyr_x[0] - shadow_area_x == 0:
                        tmap[(self.xy_map[:, :, 0] > shadow_area_x) &
                             (self.xy_map[:, :, 0] < no_shadow_area_x) &
                             (self.xy_map[:, :, 1] < no_shadow_area_y) &
                             (self.xy_map[:, :, 1] > shadow_area_y)] = pyr[2]
                    else:
                        tmap[(self.xy_map[:, :, 0] > shadow_area_x) &
                             (self.xy_map[:, :, 0] < no_shadow_area_x) &
                             (self.xy_map[:, :, 1] < no_shadow_area_y) &
                             (self.xy_map[:, :, 1] > shadow_area_y) &
                             (self.xy_map[:, :, 0] * mup + t_up < self.xy_map[:, :, 1]) &
                             (self.xy_map[:, :, 0] * mdw + t_down > self.xy_map[:, :, 1])] = pyr[2]
                else:
                    print("HÄ?")
            else:
                # print("Night")
                self.map = np.zeros_like(self.map)
            self.map -= tmap
            self.map[self.map < 0] = 0




                #shading objects:
    def add_needletree(self, frame, age, x_pos=0, y_pos=0):
        r = age*4
        h = 200 * age
        b = 50 * age

        br = int(b/2+r/2)
        h7 = int(h*0.1)
        hh7 = int((h-h7)/2)
        br4 = int(b/4+r/2)
        h7hh7 = int(h7+hh7)
        b2 = int(b/2)
        self.add_cubes_to_map(frame=frame, cubes=[[[x_pos - br,  y_pos - br,  h7],    [b,  b,  hh7], 0.65],
                                                  [[x_pos - br4, y_pos - br4, h7hh7], [b2, b2, hh7], 0.65]])
        self.add_cubes_to_map(frame=frame, cubes=[[[x_pos, y_pos, 0], [r, r, h], 0.7]])

        # self.add_cubes_to_map(frame=frame, cubes=[[[x_pos - 300, y_pos - 300, 500], [700, 700, 2400], 0.65],
        #                                           [[x_pos - 200, y_pos - 200, 2900], [500, 500, 2500], 0.65]])
        # self.add_cubes_to_map(frame=frame, cubes=[[[x_pos, y_pos, 0], [100, 100, 5500], 0.7]])
    def add_hardwoodtree(self, frame, age=20, x_pos=0, y_pos=0):
        doy = self.dates_datetime[frame].timetuple().tm_yday
        if (doy > 172) & (doy < 266):
            print("summer")
            value = 0.8
        if (doy > 266)  & (doy < 355) | (doy > 79) & (doy < 172):
            print("autumn / spring")
            value = 0.4
        if (doy > 355) & (doy < 79):
            print("winter")
            value = 0.1

        r = age*2
        h = age*15
        R = age*20
        rh = int(R/3)
        rR = int(R*0.66)
        R2 = int(R/2) - int(r/2)
        rh2 = int(rh/2)-int(r/2)

        self.add_cubes_to_map(frame=frame, cubes=[[[x_pos, y_pos, 0], [r, r, h], 0.7]])
        self.add_cubes_to_map(frame=frame, cubes=[[[x_pos - R2, y_pos - R2, h], [R, R, rh], value],
                                                  [[x_pos - rh2, y_pos - rh2, rh+h], [rR, rR, rh], value],
                                                  [[x_pos - rh2, y_pos - rh2, rh+rh+h], [rR, rR, rh], value]])
    def add_chimney(self, frame, x_pos=0,y_pos=0):
        self.add_cubes_to_map(frame, cubes=[[x_pos,y_pos, 0], [100,100,100],0.7])

    ######################### ---- CLOUDS -----------------#########################################
    def add_random_cloud(self, cloud_factor, scatter_factor = 8):
        """Adds a cloudy sky to the map.
        cloud_factor: flaot value between 0-1. Tells how much irradiation is lost due to clouds.
                        0: no clouds, 1: no light left
        scatter_factor: Clouds remove shadow. This factor tells how much the shadow is weakened."""
        omask = (self.map!=0)&(self.map!=1)
        self.map[omask] += (1-cloud_factor)*scatter_factor
        self.map[self.map>cloud_factor] = cloud_factor

    ######################### ---- SOILING -----------------#########################################
    # Generall Soiling
    def soiling_bottom_dirt(self,):
        pass



if __name__ =="__main__":
    import matplotlib.pyplot as plt

    shader = Shading(latitude=11, longitude=40, start_date="2021-09-01", end_date="2021-09-02", frequence=10)
    ########## ++++++++ TEst Tanne -------- ############
    shader.build_map(5000, 5000)
    shader.add_hardwoodtree(frame=6*12, age=25, x_pos=4000, y_pos=1000)
    plt.imshow(shader.map, vmin=0, vmax=1)
    plt.show()


    ########### ---------- Test Pyramide -------------- ############
    fig = plt.figure()
    for f in range(0,100):
        shader.build_map(400, 400)
        #smap = shader.add_pyramid_to_map(f)
        smap = shader.add_cubes_to_map(f)
        module_list = [[[10, 10], [60, 40]], [[100, 10], [60, 40]]]
        shadow_list = shader.module_shadded(smap,module_list)
        print(shadow_list)
        plt.title(str(f))
        plt.imshow(smap)#shader.map)
        for m in module_list:
            shader.plot_module(m)
        fig.canvas.draw()
        if f != 99:
            plt.pause(0.1)
            fig.clear()
    plt.show()



    ###########-----Test Clouds-----------##################
    shader.build_map(5000, 5000)
    shader.add_Tanne(4000,300)
    shader.add_Tanne(4000, 1000)
    plt.imshow(shader.map, vmin=0, vmax=1)
    plt.show()



    shader.build_map(10000,10000)
    shader.add_cubes_to_map(frame=100, cubes=[[[1000, 1000, 0], [100, 100, 5500], 0.7]])
    print(shader.map)
    #shader.add_random_cloud(5,0.92)
    plt.imshow(shader.map, vmin=0,vmax=1)
    plt.show()


    fig = plt.figure()
    for f in range(0,100):
        shader.build_map(400, 400)
        shader.add_cumulus_cloud(f,0.6)
        plt.title(str(f))
        plt.imshow(shader.map)
        fig.canvas.draw()
        if f != 99:
            plt.pause(0.1)
            fig.clear()
    plt.show()

    fig = plt.figure()
    for f in range(0,100):
        shader.build_map(400, 400)
        shader.add_stratoculmulus_cloud(f,5,100,1, windspeed=np.array([-5,3]))
        plt.title(str(f))
        plt.imshow(shader.map)
        fig.canvas.draw()
        if f != 99:
            plt.pause(0.1)
            fig.clear()
    plt.show()

    shader.build_map(400,400)
    shader.add_noise_cloudy_sky(0.7)
    plt.imshow(shader.map)
    plt.show()

    ############----Test Shadow----------###################
    shader.build_map(400,400)
    shader.add_cubes_to_map(frame=6*12, cubes=[[[50, 50, 0], [20, 20, 200], 0.3],
                                            [[30, 50, 200], [60, 20, 20], 0.3]])
    plt.imshow(shader.map, vmin=0, vmax=1)
    plt.show()






    for i in range(0,140):
        shader.build_map(900,900)
        shader.add_cubes_to_map(frame=i, cubes=[[[450,450,0],[20,20,200],0.3],
                                                [[430,450,200],[60,20,20],0.3]])
        shader.add_cubes_to_map(frame=i, cubes=[[[150, 150, 0], [60, 10, 100], 0.1],
                                                [[150, 120, 0], [10, 60, 100], 0.1]])

        shader.add_cubes_to_map(frame=i, cubes=[[[500, 150, 0], [60, 50, 400], 0.5]])
        shadow_map = shader.get_map()

        module = [[100,200],[40,60]]
        # irr = shader.get_module_irradiation(module)
        # plt.imshow(irr)
        # plt.show()
        # print(irr.shape)
        # w_irr = shader.calc_waver_mean(irr, 4,5)
        # print(w_irr.shape)
        # plt.imshow(w_irr)
        # plt.show()
        shader.get_module_irradiation(module)


        plt.title(f"Frame: {i}")
        plt.imshow(shadow_map)
        shader.plot_module(module)
        plt.axis("off")
        plt.show()