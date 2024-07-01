from datetime import datetime, timezone, timedelta
import pandas as pd
from tqdm import tqdm
import numpy as np
from pysolar import solar, radiation


class TimeAndSunAngle():
    def __init__(self, start_date="2022-07-02", end_date="2022-07-25", frequence = 10,
                 longitude=45, latitude=11):
        """Extra Class including attributes and functions which are needed by Irradiance.py and Shading.py
            Was created to reduce code and to have a more stable system.

            Can calculate a list of dates with get_dates_pandas or get_dates_datetime.
            Can calculate sun location via get_altitude or get_azimuth."""


        s = start_date.split("-")  # YYYY-MM-DD
        e = end_date.split("-")  # YYYY-MM-DD
        self.start = datetime(int(s[0]), int(s[1]), int(s[2]), tzinfo=timezone.utc)
        self.end = datetime(int(e[0]), int(e[1]), int(e[2]), tzinfo=timezone.utc)
        self.fr = frequence  # in min

        self.longitude = longitude
        self.latitude = latitude

        self.dates_datetime = None
        self.dates_pandas = None

        self.altitude = None
        self.azimuth = None

    def get_dates_pandas(self):
        """"Calculate Dates - pandas - fast version, but not always possible
        returns pandas Series of dates"""
        self.dates_pandas = pd.date_range(self.start, self.end, freq= timedelta(minutes= self.fr), tz=timezone.utc)[:-1]
        return self.dates_pandas
    def get_dates_datetime(self):
        """Calculate Dates - for loop - slow version, often possible
        returns list with datetimes"""
        self.dates_datetime = [self.start + timedelta(minutes= i * self.fr) for i in tqdm(range(int((self.end - self.start).days * 24 * 60 / self.fr)))]
        return self.dates_datetime

    def tasa_get_altitude(self):
        """Calculates sun altitude"""
        if self.dates_datetime is None: self.get_dates_datetime()
        print("TimeAndSunAngle: calculate altitude")
        self.altitude = np.deg2rad(np.array([solar.get_altitude(self.latitude, self.longitude,d) for d in tqdm(self.dates_datetime)]))
        return self.altitude

    def tasa_get_azimuth(self):
        """Calculates sun azimuth"""
        if self.dates_datetime is None: self.get_dates_datetime()
        print("TimeAndSunAngle: calculate azimuth")
        self.azimuth = np.deg2rad( np.array([solar.get_azimuth(self.latitude, self.longitude,d) for d in tqdm(self.dates_datetime)]))
        return self.azimuth