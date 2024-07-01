from .TimeAndSunAngle import TimeAndSunAngle

from solarpy import irradiance_on_plane
from pysolar import solar, radiation
from pvlib import  clearsky, atmosphere, solarposition, irradiance
from pvlib.location import Location
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


class Sun_Simulator(TimeAndSunAngle):
    def __init__(self, start_date="2022-07-02", end_date="2022-07-25", frequence = 10,
                 longitude=45, latitude=11, panel_altitude=300, tilt_angle=10, azimuth_angle=10, albedo = 0.22):
        """
        Used to simulate the sun-irradiation.  Needs TimeAndSunAngle.py to work!
        There are three different Libaries to calculate the GHI for a certain timeintervall:
            - pysolar [irradiance_pysolar]
            - solarpy [irradiance_solarpy]
            - pvlib (recommended) [irradiance_pvlib]

        Each one needs before calculating a list of dates, which is automaticlly calculated via get_dates_pandas or
        get_dates_datetime -> funcs can be found in TimeAndSunAngle.
        The GHI can be transformed to the GTI by ghi_to_gti_tiltsurface(ghi).


        :param start_date: date in YYYY-MM-DD when the simulation starts
        :param end_date:  date in YYYY-MM-DD when the simulation ends
        :param frequence: frequence in min. Each ... min a calculation is done
        :param longitude: longitude of area where irradiation has to be calculated
        :param latitude: latitude of area where irradiation has to be calculated
        :param panel_altitude: heigh over sea level of area where irradiation has to be calculated
        :param tilt_angle:  angle of module to surface
        :param azimuth_angle: angle of module to North
        :param albedo: reflected fraction of GHI.
                        0: surrounding surface is very dark,
                        1: surrounding surface is bright white or metallic or mirror
                        Examples:   Urban environment: 0.14-0.22
                                    Grass: 0.15-0.25, Fresh Grass: 0.26
                                    Fresh Snow: 0.82, Wet Snow: 0.55-0.75,
                                    Dry asphalt: 0.09-0.15,  Wet asphalt: 0.18


        """

        TimeAndSunAngle.__init__(self, start_date=start_date, end_date=end_date, frequence=frequence,
                               longitude=longitude, latitude=latitude)

        self.panel_altitude = panel_altitude
        self.tilt_anlge = np.deg2rad(tilt_angle)
        self.azimuth_angle = np.deg2rad(azimuth_angle)
        self.albedo = albedo

        ##############################
        # -- General fix constants
        self.sigma = 5.67*1e-8 #W/m2K4  stefan-bolzmann-const
        self.ineichen = None

    def irradiance_pysolar(self):
        """Calculates irradiation for each timestep with pysolar.
        returns np.array including ghi in Watt"""
        print("Calculate irradiation - pysolar")
        dates = self.get_dates_datetime()
        altitudes = [solar.get_altitude(self.latitude, self.longitude, d) for d in tqdm(dates)]
        return np.array([radiation.get_radiation_direct(d, a) for d, a in tqdm(zip(dates, altitudes))])

    def irradiance_solarpy(self, vnorm = np.array([-1,-1,-0.2])):
        """Calculates irradiation for each timestep with solarpy.
                returns np.array including ghi in Watt"""
        print("Calculate irradiation - solarpy")
        dates = self.get_dates_pandas()
        irr = [irradiance_on_plane(vnorm=vnorm, h= self.panel_altitude,date=d, lat= self.latitude) for d in tqdm(dates)]
        return np.array(irr)

    def irradiance_pvlib(self):
        """Calculates irradiation for each timestep with pvlib.
        It is advised to use this one for irradiation calculation, since this one is the most accurate one.
        returns np.array including ghi in Watt"""
        if self.dates_pandas is None:
            self.get_dates_pandas()
        print("Calculate irradiation -  pvlib")
        loc = Location(latitude=self.latitude, longitude=self.longitude)
        times = pd.date_range(start=self.start, end=self.end, freq=f'{self.fr}min', tz=loc.tz)
        solpos = solarposition.get_solarposition(times, self.latitude, self.longitude)
        apparent_zenith = solpos["apparent_zenith"]
        airmass = atmosphere.get_relative_airmass(apparent_zenith)
        pressure = atmosphere.alt2pres(self.panel_altitude)
        self.airmass = atmosphere.get_absolute_airmass(airmass, pressure)
        linke_turbidity = clearsky.lookup_linke_turbidity(times, self.latitude, self.longitude)
        dni_extra = irradiance.get_extra_radiation(times)
        ineichen = clearsky.ineichen(apparent_zenith, self.airmass, linke_turbidity, self.panel_altitude, dni_extra)
        print("irradiation calculated! - pvlib")
        self.ineichen = ineichen
        return ineichen["ghi"].values[:-1]

    def ghi_to_gti_tiltsurface(self, ghi):
        """transforms the GHI to GTI"""
        print("Calculate ghi with tiltsurface")
        daysofyear = pd.Series(self.dates_pandas).dt.dayofyear.values
        declination_angle = np.deg2rad(23.45)*np.sin((2*np.pi/365)*(284+daysofyear))
        a = np.pi/2-np.deg2rad(self.latitude)+declination_angle
        print("ghi with tiltsurface calculated")
        return ghi * np.sin(a+np.deg2rad(self.tilt_anlge))/np.sin(a)

    def irradiance_on_plane(self, sky_diffuse_model = "simple sandia", recalculate_ineichen=False, recalculate_azimuth = False, recalculate_altitude = False):
        """"""
        if (self.ineichen is None) | (recalculate_ineichen):
            if self.ineichen == None:
                print("Sun_Simulator: GHI and DNI are calculated via irradiance_pvlib.")
            self.irradiance_pvlib()
        if (self.azimuth is None) | (recalculate_azimuth): self.tasa_get_azimuth()
        if (self.altitude is None) | (recalculate_altitude):self.tasa_get_altitude()
        GHI, DNI, DHI = self.ineichen["ghi"].values[:-1], self.ineichen["dni"].values[:-1], self.ineichen["dhi"].values[:-1]
        zenith = np.pi/2 - self.altitude

        AOI = np.arccos(np.cos(zenith)*np.cos(self.tilt_anlge)+np.sin(zenith)*np.sin(self.tilt_anlge)*np.cos(self.azimuth-self.azimuth_angle))
        POA_BEAM = DNI * np.cos(AOI)
        POA_Ground_Reflection = GHI * self.albedo * (1-np.cos(self.tilt_anlge))/2

        POA_isotropic_sky_diffuse = DHI * (1 + np.cos(self.tilt_anlge)) / 2

        if sky_diffuse_model == "isotropic": POA_Sky_Diffuse = POA_isotropic_sky_diffuse
        elif sky_diffuse_model == "simple sandia": POA_Sky_Diffuse = POA_isotropic_sky_diffuse + GHI * (0.012*zenith - 0.04) *(1-np.cos(self.tilt_anlge))/2
        elif sky_diffuse_model == "Hay and Avies":
            daysofyear = pd.Series(self.dates_pandas).dt.dayofyear.values
            b = 2* np.pi * daysofyear/365
            extraterrestial_radiation = 1367 * (1.0011 + 0.034221*np.cos(b)+ 0.00128*np.sin(b) + 0.000719*np.cos(2*b)+ 0.000077*np.sin(2*b))
            anisotropy_index = DNI/extraterrestial_radiation
            POA_Sky_Diffuse = DHI * anisotropy_index * np.cos(AOI)/np.cos(zenith) + (1-anisotropy_index)*POA_isotropic_sky_diffuse
        elif sky_diffuse_model == "Reindl":
            daysofyear = pd.Series(self.dates_pandas).dt.dayofyear.values
            b = 2 * np.pi * daysofyear / 365
            extraterrestial_radiation = 1367 * (1.0011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b) + 0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b))
            anisotropy_index = DNI / extraterrestial_radiation
            POA_Sky_Diffuse = DHI * anisotropy_index * np.cos(AOI) + (1-anisotropy_index) * POA_isotropic_sky_diffuse*(1+ np.sqrt(DNI*np.cos(zenith)/(GHI+ np.finfo(np.float32).eps))*np.sin(self.tilt_anlge/2)**2)
        elif sky_diffuse_model == "Perez":

            ff = np.array([[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022],
                  [0.13, 0.683, -0.151, -0.019, 0.066, -0.029],
                  [0.33, 0.487, -0.221, 0.055, -0.064, -0.026],
                  [0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
                  [0.873, -0.392, -0.362, 0.226, -0.462, 0.001],
                  [1.132, -1.237, -0.412, 0.288, -0.823 ,0.056],
                  [1.06, -1.6, -0.359, 0.264, -1.127, 0.131],
                  [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]])

            daysofyear = pd.Series(self.dates_pandas).dt.dayofyear.values
            b = 2 * np.pi * daysofyear / 365
            extraterrestial_radiation = 1367 * (1.0011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b) + 0.000719 * np.cos(
                2 * b) + 0.000077 * np.sin(2 * b))

            clearness = ((DHI+DNI+ np.finfo(np.float32).eps)/(DHI+ np.finfo(np.float32).eps) + 1.041*zenith**3 )/(1+ 1.041*zenith**3)
            clearness[clearness>7]=7

            f = ff[np.array(clearness, dtype=np.int32)]
            D = DHI * self.airmass[:-1] / extraterrestial_radiation
            F1 = f[:,0]+f[:,1]*D + zenith*f[:,2]
            F1[F1<=0] = 0
            F2 = f[:,3] + f[:,4]*D + zenith*f[:,5]
            a= np.cos(AOI)
            a[a<0] = 0
            b = np.cos(zenith)
            b[b<np.cos(np.deg2rad(85))] = np.cos(np.deg2rad(85))
            POA_Sky_Diffuse = DHI * ( (1-F1)*(1+np.cos(self.tilt_anlge))/2 + F1*(a/b) + F2*np.sin(self.tilt_anlge) )

        POA = POA_BEAM + POA_Ground_Reflection + POA_Sky_Diffuse
        POA[POA<0] = 0

        self.irradiance = POA
        return POA
