from pathlib import Path
import numpy as np
import xarray as xr

from config import DEFAULT_DLAT, DEFAULT_DLON, GRID_PARAMS_1D, GRID_PARAMS_2D, ALL_PARAMS

PARAMS_ALLOW_NEGATIVE = {"temperature", "temperature_slwc"}
PARAMS_EXCLUDE_ZERO   = {"slwp", "slwc"}
PARAMS_NO_FILTER      = {"land_water_flag"}


class GridAccumulator:

    def __init__(self,
                 dlat: float = DEFAULT_DLAT,
                 dlon: float = DEFAULT_DLON,
                 lat_range: tuple = (-90.0, 90.0),
                 lon_range: tuple = (-180.0, 180.0)):

        self.dlat = dlat
        self.dlon = dlon

        self.lat_bins = np.arange(lat_range[0], lat_range[1] + dlat, dlat)
        self.lon_bins = np.arange(lon_range[0], lon_range[1] + dlon, dlon)
        self.n_lat = len(self.lat_bins)
        self.n_lon = len(self.lon_bins)

        self._days: dict[str, dict] = {}

    @property
    def dates(self) -> list[str]:
        return sorted(self._days.keys())

    @property
    def n_orbits(self) -> int:
        return sum(self._days[d]["_n_orbits"] for d in self._days)

    def n_orbits_range(self, d1: str, d2: str):
        days_range = [d for d in self.dates if d1 <= d <= d2]
        if not days_range:
            raise ValueError(f"Aucun jour en {d1} et {d2}")
        result = {d: self._days[d]["_n_orbits"] for d in days_range}
        total = sum(result.values())
        return total

    def _init_day(self, day_str: str) -> None:
        shape = (self.n_lat, self.n_lon)
        self._days[day_str] = {
            "_n_orbits": 0,
            **{
                p: {
                    "sum":   np.zeros(shape),
                    "sum2":  np.zeros(shape),
                    "count": np.zeros(shape, dtype=np.int32),
                }
                for p in ALL_PARAMS
            }
        }

    def _cell_indices(self, lat_arr, lon_arr):
        lat = np.asarray(lat_arr, dtype=float)
        lon = np.asarray(lon_arr, dtype=float)
        lat_safe = np.where(np.isnan(lat) | np.isinf(lat), self.lat_bins[0], lat)
        lon_safe = np.where(np.isnan(lon) | np.isinf(lon), self.lon_bins[0], lon)

        with np.errstate(invalid="ignore", divide="ignore"):
            i_lat = np.floor((lat_safe - self.lat_bins[0]) / self.dlat)
            i_lon = np.floor((lon_safe - self.lon_bins[0]) / self.dlon)

        i_lat = np.nan_to_num(i_lat, nan=0, posinf=0, neginf=0).astype(int)
        i_lon = np.nan_to_num(i_lon, nan=0, posinf=0, neginf=0).astype(int)
        i_lat = np.clip(i_lat, 0, self.n_lat - 1)
        i_lon = np.clip(i_lon, 0, self.n_lon - 1)
        return i_lat, i_lon

    @staticmethod
    def _column_mean(arr2d):
        with np.errstate(all="ignore"):
            result = np.nanmean(arr2d, axis=1)
        result[np.all(np.isnan(arr2d), axis=1)] = np.nan
        return result

    @staticmethod
    def _dominant_particle(arr2d):
        n_time = arr2d.shape[0]
        result = np.zeros(n_time, dtype=float)
        for i in range(n_time):
            col = arr2d[i, :]
            valid = col[(col >= 1) & (col <= 13)]
            if len(valid) > 0:
                vals, counts = np.unique(valid, return_counts=True)
                result[i] = vals[np.argmax(counts)]
        return result

    @staticmethod
    def _compute_slwc_slwp(orbit_data: dict) -> dict:
        """
        Calcule SLWC (2D), SLWP (1D), temperature_slwc (1D) et cloud_level_slwc (1D).
        - SLWC       = LWC restreint aux niveaux où T <= 0°C
        - SLWP       = intégrale verticale de SLWC par rectangles (dz=100m)
                       uniquement sur les niveaux avec altitude valide
        - temperature_slwc  = température moyenne là où SLWC > 0 ET altitude valide
        - cloud_level_slwc  = altitude moyenne là où SLWC > 0 ET altitude valide
        La température doit être déjà en °C.
        """
        lwc         = orbit_data.get("lwc")
        temperature = orbit_data.get("temperature")  # déjà en °C
        height      = orbit_data.get("height")

        if lwc is None or temperature is None or height is None:
            return orbit_data

        orbit_data = dict(orbit_data)

        lwc_arr    = np.asarray(lwc, dtype=float)
        temp_arr   = np.asarray(temperature, dtype=float)
        height_arr = np.asarray(height, dtype=float)

        # Masquage des fill values du LWC
        lwc_clean = np.ma.masked_where(
            (lwc_arr < 0) | (lwc_arr < -9000), lwc_arr
        )

        # Masquage des fill values de la température (déjà en °C)
        temp_clean = np.ma.masked_where(
            (np.isnan(temp_arr)) | (temp_arr < -200), temp_arr
        )

        # SLWC : LWC restreint à T <= 0°C
        slwc = np.where(temp_clean <= 0, lwc_clean, np.nan)
        slwc = np.ma.masked_where(slwc < -900, slwc)
        orbit_data["slwc"] = slwc

        # Niveaux avec altitude valide (même logique que slwc_profile[valid])
        valid_height = ~np.isnan(height_arr)  # (n_pixels, 206)

        # SLWP : intégration verticale par rectangles (dz=100m)
        # uniquement sur les niveaux avec altitude valide
        slwc_filled = np.where(np.ma.getmaskarray(slwc), 0.0, np.ma.filled(slwc, 0.0))
        slwc_clean  = np.where(valid_height, slwc_filled, 0.0)
        slwp        = np.nansum(slwc_clean, axis=1) * 100
        orbit_data["slwp"] = slwp

        # temperature_slwc et cloud_level_slwc
        # uniquement là où SLWC > 0 ET altitude valide
        n_pixels    = slwc_filled.shape[0]
        temp_slwc   = np.full(n_pixels, np.nan)
        cloud_level = np.full(n_pixels, np.nan)
        temp_filled = np.ma.filled(temp_clean, np.nan)

        for k in range(n_pixels):
            valid_k   = valid_height[k, :]                        # altitude valide
            mask_slwc = (slwc_filled[k, :] > 0) & valid_k        # SLWC > 0 ET altitude valide

            if np.sum(mask_slwc) > 0:
                temp_slwc[k]   = np.nanmean(temp_filled[k, mask_slwc])
                cloud_level[k] = np.nanmean(height_arr[k, mask_slwc])

        orbit_data["temperature_slwc"] = temp_slwc
        orbit_data["cloud_level_slwc"] = cloud_level

        return orbit_data

    @staticmethod
    def _clean_values(arr: np.ndarray, param: str) -> np.ndarray:
        if param in PARAMS_NO_FILTER:
            return np.where(np.isnan(arr), np.nan, arr)
        elif param in PARAMS_ALLOW_NEGATIVE:
            return np.where((np.isnan(arr)) | (arr < -200), np.nan, arr)
        elif param in PARAMS_EXCLUDE_ZERO:
            return np.where((arr <= 0) | np.isnan(arr), np.nan, arr)
        else:
            return np.where(arr < 0, np.nan, arr)

    def accumulate(self, orbit_data: dict) -> None:

        t0 = orbit_data.get("start_time")
        if t0 is None:
            print("start_time absent orbite ignorée.")
            return

        day_str = t0.decode().replace("UTC=", "").replace("Z", "").split("T")[0]
        print(day_str)
        if day_str not in self._days:
            self._init_day(day_str)

        day = self._days[day_str]

        lat = np.asarray(orbit_data["lat"], dtype=float)
        lon = np.asarray(orbit_data["lon"], dtype=float)
        i_lat, i_lon = self._cell_indices(lat, lon)
        invalid = np.isnan(lat) | np.isnan(lon)

        # Conversion Kelvin → Celsius
        orbit_data = dict(orbit_data)
        if "temperature" in orbit_data:
            temp = np.asarray(orbit_data["temperature"], dtype=float)
            temp = np.where(
                (np.isnan(temp)) | (temp < 150) | (temp > 400),
                np.nan, temp
            )
            temp = temp - 273.15
            orbit_data["temperature"] = temp

        # Calcul de SLWC / SLWP / temperature_slwc / cloud_level_slwc
        orbit_data = self._compute_slwc_slwp(orbit_data)

        # Accumulation des variables 1D
        for param in GRID_PARAMS_1D:
            raw = orbit_data.get(param)
            if raw is None:
                continue
            arr    = np.asarray(raw, dtype=float)
            values = self._clean_values(arr, param)
            values[invalid] = np.nan
            _add_at(day[param], i_lat, i_lon, values)

        # Accumulation des variables 2D
        for param in GRID_PARAMS_2D:
            raw = orbit_data.get(param)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float)
            if param == "particle_type":
                values = self._dominant_particle(arr)
            else:
                cleaned = self._clean_values(arr, param)
                values  = self._column_mean(cleaned)
            values[invalid] = np.nan
            _add_at(day[param], i_lat, i_lon, values)

        day["_n_orbits"] += 1

    def mean(self, day: str) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent. Disponibles : {self.dates}")
        return _compute_mean(self._days[day])

    def std(self, day: str) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent. Disponibles : {self.dates}")
        return _compute_std(self._days[day])

    def count(self, day: str) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent. Disponibles : {self.dates}")
        return {p: self._days[day][p]["count"].copy() for p in ALL_PARAMS}

    def mean_range(self, d1: str, d2: str) -> dict:
        merged = self._merge_days(d1, d2)
        return _compute_mean(merged)

    def std_range(self, d1: str, d2: str) -> dict:
        merged = self._merge_days(d1, d2)
        return _compute_std(merged)

    def count_range(self, d1: str, d2: str) -> dict:
        merged = self._merge_days(d1, d2)
        return {p: merged[p]["count"].copy() for p in ALL_PARAMS}

    def _merge_days(self, d1: str, d2: str) -> dict:
        days_in_range = [d for d in self.dates if d1 <= d <= d2]
        if not days_in_range:
            raise ValueError(
                f"Aucun jour entre {d1} et {d2}. "
                f"Disponibles : {self.dates[0]} -> {self.dates[-1]}"
            )

        shape = (self.n_lat, self.n_lon)
        merged = {
            p: {
                "sum":   np.zeros(shape),
                "sum2":  np.zeros(shape),
                "count": np.zeros(shape, dtype=np.int32),
            }
            for p in ALL_PARAMS
        }
        for d in days_in_range:
            for p in ALL_PARAMS:
                merged[p]["sum"]   += self._days[d][p]["sum"]
                merged[p]["sum2"]  += self._days[d][p]["sum2"]
                merged[p]["count"] += self._days[d][p]["count"]

        return merged

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        ds = xr.Dataset(
            coords={
                "lat": ("lat", self.lat_bins,
                        {"units": "degrees_north", "long_name": "latitude"}),
                "lon": ("lon", self.lon_bins,
                        {"units": "degrees_east",  "long_name": "longitude"}),
            },
            attrs={
                "title": "EarthCARE ACM_CLP_2B daily gridded statistics",
                "dlat": self.dlat,
                "dlon": self.dlon,
                "n_days": len(self._days),
                "n_orbits": self.n_orbits,
                "date_start": self.dates[0] if self.dates else "",
                "date_end": self.dates[-1] if self.dates else "",
                "conventions": "CF-1.8",
            },
        )

        for day_str, day_data in sorted(self._days.items()):
            safe = day_str.replace("-", "")
            ds[f"n_orbits_{safe}"] = int(day_data["_n_orbits"])
            for p in ALL_PARAMS:
                ds[f"{safe}_{p}_sum"]   = (["lat", "lon"], day_data[p]["sum"])
                ds[f"{safe}_{p}_sum2"]  = (["lat", "lon"], day_data[p]["sum2"])
                ds[f"{safe}_{p}_count"] = (["lat", "lon"], day_data[p]["count"].astype(float))

        ds.to_netcdf(filepath, mode="w")
        size_mb = Path(filepath).stat().st_size / 1e6
        print(f"Sauvegardé {filepath}")
        print(f"{len(self._days)} jours | {self.n_orbits} orbites | {size_mb:.1f} MB")
        if self.dates:
            print(f"Période : {self.dates[0]} {self.dates[-1]}")

    @classmethod
    def load(cls, filepath: str) -> "GridAccumulator":
        ds = xr.open_dataset(filepath)

        g = cls.__new__(cls)
        g.dlat = float(ds.attrs["dlat"])
        g.dlon = float(ds.attrs["dlon"])
        g.lat_bins = ds["lat"].values
        g.lon_bins = ds["lon"].values
        g.n_lat = len(g.lat_bins)
        g.n_lon = len(g.lon_bins)
        g._days = {}

        day_keys = [k[9:] for k in ds.data_vars if k.startswith("n_orbits_")]

        for safe in sorted(day_keys):
            day_str = f"{safe[:4]}-{safe[4:6]}-{safe[6:]}"
            day_data = {"_n_orbits": int(ds[f"n_orbits_{safe}"].values)}
            for p in ALL_PARAMS:
                day_data[p] = {
                    "sum":   ds[f"{safe}_{p}_sum"].values.copy(),
                    "sum2":  ds[f"{safe}_{p}_sum2"].values.copy(),
                    "count": ds[f"{safe}_{p}_count"].values.astype(np.int32).copy(),
                }
            g._days[day_str] = day_data

        ds.close()
        print(f"Charge : {filepath}")
        if g.dates:
            print(f"{len(g._days)} jours | {g.n_orbits} orbites | "
                  f"{g.dates[0]} to {g.dates[-1]}")
        return g

    def __repr__(self):
        period = f"{self.dates[0]} to {self.dates[-1]}" if self.dates else "vide"
        return (
            f"GridAccumulator("
            f"dlat={self.dlat}°, dlon={self.dlon}°, "
            f"{self.n_lat}×{self.n_lon} cells, "
            f"{len(self._days)} jours, "
            f"{self.n_orbits} orbites, "
            f"{period})"
        )


def _add_at(param_dict: dict, i_lat, i_lon, values) -> None:
    valid = ~np.isnan(values)
    if not np.any(valid):
        return
    v  = values[valid]
    il = i_lat[valid]
    jl = i_lon[valid]
    np.add.at(param_dict["sum"],   (il, jl), v)
    np.add.at(param_dict["sum2"],  (il, jl), v ** 2)
    np.add.at(param_dict["count"], (il, jl), 1)


def _compute_mean(day_data: dict) -> dict:
    result = {}
    for p in ALL_PARAMS:
        n = day_data[p]["count"].astype(float)
        s = day_data[p]["sum"]
        with np.errstate(invalid="ignore", divide="ignore"):
            result[p] = np.where(n > 0, s / n, np.nan)
    return result


def _compute_std(day_data: dict) -> dict:
    result = {}
    for p in ALL_PARAMS:
        n  = day_data[p]["count"].astype(float)
        s  = day_data[p]["sum"]
        s2 = day_data[p]["sum2"]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(
                n >= 2,
                (s2 - s ** 2 / n) / (n - 1),
                np.nan,
            )
            result[p] = np.sqrt(np.maximum(var, 0.0))
    return result