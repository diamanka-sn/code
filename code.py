def plot_cloud_height_distribution(grid, day=None, d1=None, d2=None,
                                    bins=50,
                                    xlim=(0, 5000),
                                    mask_type: str = "all",
                                    title=None):
    """
    Distribution de la hauteur des nuages SLW au-dessus de la surface
    height = cloud_level_slwc - surface_elevation
    depuis le gridding.
    """
    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)

    cloud_level       = means["cloud_level_slwc"]
    surface_elevation = means["surface_elevation"]
    flag              = means["land_water_flag"]

    # Masque spatial
    if mask_type == "land":
        spatial_mask = np.round(flag) == 1
    elif mask_type == "ocean":
        spatial_mask = np.round(flag) == 0
    else:
        spatial_mask = np.ones(cloud_level.shape, dtype=bool)

    # Hauteur au-dessus de la surface
    height = cloud_level - surface_elevation

    # Filtrer
    valid  = spatial_mask & np.isfinite(height) & (height >= 0)
    height = height[valid].ravel()

    print(f"N cellules valides : {len(height)}")
    print(f"Height min/max : {height.min():.1f} / {height.max():.1f} m")
    print(f"Height moyenne : {np.mean(height):.1f} m")
    print(f"Height médiane : {np.median(height):.1f} m")

    if len(height) == 0:
        raise ValueError("Aucune valeur valide.")

    # Histogramme
    counts, edges = np.histogram(height, bins=bins, range=xlim)
    centers       = 0.5 * (edges[:-1] + edges[1:])

    mask_label = {"all": "Land + Ocean", "land": "Land only",
                  "ocean": "Ocean only"}.get(mask_type, mask_type)

    # Figure
    fig, ax = plt.subplots(figsize=(6, 8))
    fig.patch.set_facecolor("white")

    ax.plot(counts, centers, 'b-', linewidth=2)

    ax.axhline(np.mean(height),   color='r', linewidth=1.5,
               linestyle='--', label=f'Mean : {np.mean(height):.0f} m')
    ax.axhline(np.median(height), color='g', linewidth=1.5,
               linestyle='--', label=f'Median : {np.median(height):.0f} m')

    ax.set_ylabel("Height above surface (m)", fontsize=11)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_ylim(xlim)
    ax.set_xlim(left=0)
    ax.tick_params(top=True, right=True, which="both", direction="in")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.legend(fontsize=9)
    ax.set_title(title or f"SLW Cloud Height above Surface\n"
                          f"{n_orbits} orbits | {mask_label} | {label}\n"
                          f"N={len(height)} cells",
                 fontsize=11)

    plt.tight_layout()
    plt.show()

    return height


# Figure
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")

ax.plot(centers, counts, 'b-', linewidth=2)

ax.axvline(np.mean(height),   color='r', linewidth=1.5,
           linestyle='--', label=f'Mean : {np.mean(height):.0f} m')
ax.axvline(np.median(height), color='g', linewidth=1.5,
           linestyle='--', label=f'Median : {np.median(height):.0f} m')

ax.set_xlabel("Height above surface (m)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_xlim(xlim)
ax.set_ylim(bottom=0)
ax.tick_params(top=True, right=True, which="both", direction="in")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.legend(fontsize=9)

plot_cloud_height_distribution(grid, mask_type="all", d1="2025-12-01", d2="2025-12-31")

# Terre uniquement
plot_cloud_height_distribution(grid, mask_type="land", d1="2025-12-01", d2="2025-12-31")

# Océan uniquement
plot_cloud_height_distribution(grid, mask_type="ocean", d1="2025-12-01", d2="2025-12-31")



def plot_slwp_temp_pd_with_mean(grid, day=None, d1=None, d2=None,
                                 bins_x=50, bins_y=50,
                                 xlim=(0, 20), ylim=(-45, 0),
                                 area_weighted=True, normalize=True,
                                 vmin=None, vmax=None, cmap="rainbow",
                                 log_density=False,
                                 mask_type: str = "all",
                                 bins_mean=15,
                                 xlabel=None, ylabel=None,
                                 title=None):
  
    means, stds, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)

    data_x = means["slwp"]
    data_y = means["temperature_slwc"]
    flag   = means["land_water_flag"]

    lon_bins = np.asarray(grid.lon_bins)
    lat_bins = np.asarray(grid.lat_bins)
    lon_c = 0.5 * (lon_bins[:-1] + lon_bins[1:]) if len(lon_bins) == data_x.shape[1] + 1 else lon_bins
    lat_c = 0.5 * (lat_bins[:-1] + lat_bins[1:]) if len(lat_bins) == data_x.shape[0] + 1 else lat_bins
    LAT2D = np.meshgrid(lon_c, lat_c)[1]

    dlat = getattr(grid, "dlat", np.median(np.diff(lat_c)))
    dlon = getattr(grid, "dlon", np.median(np.diff(lon_c)))
    cell_area = np.cos(np.deg2rad(LAT2D)) * np.deg2rad(dlat) * np.deg2rad(dlon)

    if mask_type == "land":
        spatial_mask = np.round(flag) == 1
    elif mask_type == "ocean":
        spatial_mask = np.round(flag) == 0
    else:
        spatial_mask = np.ones(data_x.shape, dtype=bool)

    valid = spatial_mask & np.isfinite(data_x) & np.isfinite(data_y)
    x = data_x[valid].ravel()   
    y = data_y[valid].ravel() 
    w = cell_area[valid].ravel() if area_weighted else None

    if x.size == 0:
        raise ValueError("Aucun point valide après masquage.")

    print(f"N points valides : {x.size}")
    print(f"SLWP min/max : {x.min():.4f} / {x.max():.4f} g/m²")
    print(f"Temp min/max : {y.min():.2f} / {y.max():.2f} °C")

    y_range= ylim if ylim is not None else [y.min(), y.max()]
    hist_range = [xlim, y_range]
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y],
                                        range=hist_range, weights=w)
    if normalize and H.sum() > 0:
        H = H / H.sum() * 100
    H_plot = np.where(H.T > 0, H.T, np.nan)

    temp_edges   = np.linspace(ylim[0], ylim[1], bins_mean + 1)
    temp_centers = 0.5 * (temp_edges[:-1] + temp_edges[1:])

    bin_means= []
    bin_stds= []
    bin_centers_valid = []

    for k in range(len(temp_edges) - 1):
        mask_bin = (y >= temp_edges[k]) & (y < temp_edges[k + 1])
        if np.sum(mask_bin) >= 2:
            bin_means.append(np.mean(x[mask_bin]))   
            bin_stds.append(np.std(x[mask_bin]))    
            bin_centers_valid.append(temp_centers[k])

    bin_means= np.array(bin_means)
    bin_stds= np.array(bin_stds)
    bin_centers_valid = np.array(bin_centers_valid)

    if ylabel is None:
        ylabel = "In-Cloud SLW Temperature (°C)"
    if xlabel is None:
        xlabel = "SLWP ($g/m²$)"

    mask_label = {"all": "Land + Ocean", "land": "Land only",
                  "ocean": "Ocean only"}.get(mask_type, mask_type)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    cmap_obj = plt.get_cmap(cmap)
    lo = vmin if vmin is not None else 0
    hi = vmax if vmax is not None else np.nanmax(H_plot)

    if log_density:
        norm = mcolors.LogNorm(vmin=vmin or 1e-3, vmax=vmax or hi)
    else:
        norm = mcolors.Normalize(vmin=lo, vmax=hi)

    pc= ax.pcolormesh(xedges, yedges, H_plot, cmap=cmap_obj,
                         norm=norm, shading="auto")
    cbar = plt.colorbar(pc, ax=ax)
    cbar.set_label("Probability density (%)" if normalize else "Density", fontsize=10)

    ax.errorbar(bin_means, bin_centers_valid,
                xerr=bin_stds,
                yerr=None,
                fmt='r*', markersize=8,
                ecolor='red', elinewidth=1.5,
                capsize=3, capthick=1.5,
                label='Mean SLWP ± std per temperature bin')

    _add_ricaud_curve(ax)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(top=True, right=True, which="both", direction="in")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.legend(fontsize=9)

    ax.set_title(title or f"SLWP vs In-Cloud SLW Temperature\n"
                          f"{n_orbits} orbits | grid {dlat}°×{dlon}° | {mask_label} | {label}",
                 fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    plt.tight_layout()
    plt.show()


from pathlib import Path
from datetime import datetime, timedelta
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
        total  = sum(result.values())
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
            },
            # Par heure UTC
            "hourly_utc": {
                h: {
                    p: {
                        "sum":   np.zeros(shape),
                        "sum2":  np.zeros(shape),
                        "count": np.zeros(shape, dtype=np.int32),
                    }
                    for p in ALL_PARAMS
                }
                for h in range(24)
            },
            # Par heure locale
            "hourly_local": {
                h: {
                    p: {
                        "sum":   np.zeros(shape),
                        "sum2":  np.zeros(shape),
                        "count": np.zeros(shape, dtype=np.int32),
                    }
                    for p in ALL_PARAMS
                }
                for h in range(24)
            },
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
            col   = arr2d[i, :]
            valid = col[(col >= 1) & (col <= 13)]
            if len(valid) > 0:
                vals, counts = np.unique(valid, return_counts=True)
                result[i] = vals[np.argmax(counts)]
        return result

    @staticmethod
    def _compute_slwc_slwp(orbit_data: dict) -> dict:
        lwc         = orbit_data.get("lwc")
        temperature = orbit_data.get("temperature")
        height      = orbit_data.get("height")

        if lwc is None or temperature is None or height is None:
            return orbit_data

        orbit_data = dict(orbit_data)

        lwc_arr    = np.asarray(lwc,         dtype=float)
        temp_arr   = np.asarray(temperature, dtype=float)
        height_arr = np.asarray(height,      dtype=float)

        lwc_clean = np.ma.masked_where(
            (lwc_arr < 0) | (lwc_arr < -9000), lwc_arr
        )
        temp_clean = np.ma.masked_where(
            (np.isnan(temp_arr)) | (temp_arr < -200), temp_arr
        )

        slwc = np.where(temp_clean <= 0, lwc_clean, np.nan)
        slwc = np.ma.masked_where(slwc < -900, slwc)
        orbit_data["slwc"] = slwc

        valid_height = ~np.isnan(height_arr)
        slwc_filled  = np.where(np.ma.getmaskarray(slwc), 0.0, np.ma.filled(slwc, 0.0))
        slwc_clean_h = np.where(valid_height, slwc_filled, 0.0)
        slwp         = np.nansum(slwc_clean_h, axis=1) * 100
        orbit_data["slwp"] = slwp

        n_pixels    = slwc_filled.shape[0]
        temp_slwc   = np.full(n_pixels, np.nan)
        cloud_level = np.full(n_pixels, np.nan)
        temp_filled = np.ma.filled(temp_clean, np.nan)

        for k in range(n_pixels):
            valid_k   = valid_height[k, :]
            mask_slwc = (slwc_filled[k, :] > 0) & valid_k
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
        elif param == "surface_elevation":
            return np.where((np.isnan(arr)) | (arr < -9000), 0.0, arr)
        else:
            return np.where(arr < 0, np.nan, arr)

    @staticmethod
    def _accumulate_params(day_dict, orbit_data, i_lat, i_lon, mask=None):
        """Accumule les variables 1D et 2D dans day_dict.
        Si mask est fourni, seuls les pixels mask==True sont accumulés.
        """
        if mask is not None:
            i_lat_m = i_lat[mask]
            i_lon_m = i_lon[mask]
        else:
            i_lat_m = i_lat
            i_lon_m = i_lon

        for param in GRID_PARAMS_1D:
            raw = orbit_data.get(param)
            if raw is None:
                continue
            arr    = np.asarray(raw, dtype=float)
            values = GridAccumulator._clean_values(arr, param)
            if mask is not None:
                values = values[mask]
            _add_at(day_dict[param], i_lat_m, i_lon_m, values)

        for param in GRID_PARAMS_2D:
            raw = orbit_data.get(param)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float)
            if param == "particle_type":
                values = GridAccumulator._dominant_particle(arr)
            else:
                cleaned = GridAccumulator._clean_values(arr, param)
                values  = GridAccumulator._column_mean(cleaned)
            if mask is not None:
                values = values[mask]
            _add_at(day_dict[param], i_lat_m, i_lon_m, values)

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

        lat    = np.asarray(orbit_data["lat"],  dtype=float)
        lon    = np.asarray(orbit_data["lon"],  dtype=float)
        time_s = np.asarray(orbit_data["time"], dtype=float)
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

        # Calcul SLWC / SLWP / temperature_slwc / cloud_level_slwc
        orbit_data = self._compute_slwc_slwp(orbit_data)

        # Calcul heure UTC et locale
        t0_utc = datetime.strptime(
            t0.decode().replace("UTC=", "").replace("Z", ""), "%Y-%m-%dT%H:%M:%S"
        )

        hour_utc = np.array([
            (t0_utc + timedelta(seconds=float(s))).hour
            for s in time_s
        ])

        hour_local = np.array([
            (t0_utc + timedelta(seconds=float(s) + float(lo) * 240)).hour
            for s, lo in zip(time_s, lon)
        ])

        # --- Accumulation globale (toutes heures) ---
        self._accumulate_params(day, orbit_data, i_lat, i_lon,
                                mask=~invalid)

        # --- Accumulation par heure UTC ---
        for h in range(24):
            mask_h = (hour_utc == h) & ~invalid
            if not np.any(mask_h):
                continue
            self._accumulate_params(day["hourly_utc"][h], orbit_data,
                                    i_lat, i_lon, mask=mask_h)

        # --- Accumulation par heure locale ---
        for h in range(24):
            mask_h = (hour_local == h) & ~invalid
            if not np.any(mask_h):
                continue
            self._accumulate_params(day["hourly_local"][h], orbit_data,
                                    i_lat, i_lon, mask=mask_h)

        day["_n_orbits"] += 1

    # --- Accès aux moyennes ---

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

    # --- Accès horaire UTC ---

    def mean_hourly_utc(self, day: str, hour: int) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent.")
        return _compute_mean(self._days[day]["hourly_utc"][hour])

    def std_hourly_utc(self, day: str, hour: int) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent.")
        return _compute_std(self._days[day]["hourly_utc"][hour])

    def count_hourly_utc(self, day: str, hour: int) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent.")
        return {p: self._days[day]["hourly_utc"][hour][p]["count"].copy() for p in ALL_PARAMS}

    def mean_hourly_utc_range(self, d1: str, d2: str, hour: int) -> dict:
        merged = self._merge_hourly(d1, d2, hour, "hourly_utc")
        return _compute_mean(merged)

    def std_hourly_utc_range(self, d1: str, d2: str, hour: int) -> dict:
        merged = self._merge_hourly(d1, d2, hour, "hourly_utc")
        return _compute_std(merged)

    # --- Accès horaire local ---

    def mean_hourly_local(self, day: str, hour: int) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent.")
        return _compute_mean(self._days[day]["hourly_local"][hour])

    def std_hourly_local(self, day: str, hour: int) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent.")
        return _compute_std(self._days[day]["hourly_local"][hour])

    def count_hourly_local(self, day: str, hour: int) -> dict:
        if day not in self._days:
            raise KeyError(f"Jour '{day}' absent.")
        return {p: self._days[day]["hourly_local"][hour][p]["count"].copy() for p in ALL_PARAMS}

    def mean_hourly_local_range(self, d1: str, d2: str, hour: int) -> dict:
        merged = self._merge_hourly(d1, d2, hour, "hourly_local")
        return _compute_mean(merged)

    def std_hourly_local_range(self, d1: str, d2: str, hour: int) -> dict:
        merged = self._merge_hourly(d1, d2, hour, "hourly_local")
        return _compute_std(merged)

    # --- Fusion des jours ---

    def _merge_days(self, d1: str, d2: str) -> dict:
        days_in_range = [d for d in self.dates if d1 <= d <= d2]
        if not days_in_range:
            raise ValueError(
                f"Aucun jour entre {d1} et {d2}. "
                f"Disponibles : {self.dates[0]} -> {self.dates[-1]}"
            )
        shape  = (self.n_lat, self.n_lon)
        merged = {
            p: {"sum": np.zeros(shape), "sum2": np.zeros(shape),
                "count": np.zeros(shape, dtype=np.int32)}
            for p in ALL_PARAMS
        }
        for d in days_in_range:
            for p in ALL_PARAMS:
                merged[p]["sum"]   += self._days[d][p]["sum"]
                merged[p]["sum2"]  += self._days[d][p]["sum2"]
                merged[p]["count"] += self._days[d][p]["count"]
        return merged

    def _merge_hourly(self, d1: str, d2: str, hour: int, key: str) -> dict:
        days_in_range = [d for d in self.dates if d1 <= d <= d2]
        if not days_in_range:
            raise ValueError(f"Aucun jour entre {d1} et {d2}.")
        shape  = (self.n_lat, self.n_lon)
        merged = {
            p: {"sum": np.zeros(shape), "sum2": np.zeros(shape),
                "count": np.zeros(shape, dtype=np.int32)}
            for p in ALL_PARAMS
        }
        for d in days_in_range:
            for p in ALL_PARAMS:
                merged[p]["sum"]   += self._days[d][key][hour][p]["sum"]
                merged[p]["sum2"]  += self._days[d][key][hour][p]["sum2"]
                merged[p]["count"] += self._days[d][key][hour][p]["count"]
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
                "title":      "EarthCARE ACM_CLP_2B daily gridded statistics",
                "dlat":       self.dlat,
                "dlon":       self.dlon,
                "n_days":     len(self._days),
                "n_orbits":   self.n_orbits,
                "date_start": self.dates[0] if self.dates else "",
                "date_end":   self.dates[-1] if self.dates else "",
                "conventions": "CF-1.8",
            },
        )

        for day_str, day_data in sorted(self._days.items()):
            safe = day_str.replace("-", "")
            ds[f"n_orbits_{safe}"] = int(day_data["_n_orbits"])

            # Variables globales
            for p in ALL_PARAMS:
                ds[f"{safe}_{p}_sum"]   = (["lat", "lon"], day_data[p]["sum"])
                ds[f"{safe}_{p}_sum2"]  = (["lat", "lon"], day_data[p]["sum2"])
                ds[f"{safe}_{p}_count"] = (["lat", "lon"], day_data[p]["count"].astype(float))

            # Variables horaires UTC
            for h in range(24):
                for p in ALL_PARAMS:
                    ds[f"{safe}_h{h:02d}utc_{p}_sum"]   = (["lat", "lon"], day_data["hourly_utc"][h][p]["sum"])
                    ds[f"{safe}_h{h:02d}utc_{p}_sum2"]  = (["lat", "lon"], day_data["hourly_utc"][h][p]["sum2"])
                    ds[f"{safe}_h{h:02d}utc_{p}_count"] = (["lat", "lon"], day_data["hourly_utc"][h][p]["count"].astype(float))

            # Variables horaires locales
            for h in range(24):
                for p in ALL_PARAMS:
                    ds[f"{safe}_h{h:02d}loc_{p}_sum"]   = (["lat", "lon"], day_data["hourly_local"][h][p]["sum"])
                    ds[f"{safe}_h{h:02d}loc_{p}_sum2"]  = (["lat", "lon"], day_data["hourly_local"][h][p]["sum2"])
                    ds[f"{safe}_h{h:02d}loc_{p}_count"] = (["lat", "lon"], day_data["hourly_local"][h][p]["count"].astype(float))

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
        g.dlat     = float(ds.attrs["dlat"])
        g.dlon     = float(ds.attrs["dlon"])
        g.lat_bins = ds["lat"].values
        g.lon_bins = ds["lon"].values
        g.n_lat    = len(g.lat_bins)
        g.n_lon    = len(g.lon_bins)
        g._days    = {}

        day_keys = [k[9:] for k in ds.data_vars if k.startswith("n_orbits_")]

        for safe in sorted(day_keys):
            day_str  = f"{safe[:4]}-{safe[4:6]}-{safe[6:]}"
            shape    = (g.n_lat, g.n_lon)
            day_data = {"_n_orbits": int(ds[f"n_orbits_{safe}"].values)}

            # Variables globales
            for p in ALL_PARAMS:
                day_data[p] = {
                    "sum":   ds[f"{safe}_{p}_sum"].values.copy(),
                    "sum2":  ds[f"{safe}_{p}_sum2"].values.copy(),
                    "count": ds[f"{safe}_{p}_count"].values.astype(np.int32).copy(),
                }

            # Variables horaires UTC
            day_data["hourly_utc"] = {}
            for h in range(24):
                day_data["hourly_utc"][h] = {}
                for p in ALL_PARAMS:
                    day_data["hourly_utc"][h][p] = {
                        "sum":   ds[f"{safe}_h{h:02d}utc_{p}_sum"].values.copy(),
                        "sum2":  ds[f"{safe}_h{h:02d}utc_{p}_sum2"].values.copy(),
                        "count": ds[f"{safe}_h{h:02d}utc_{p}_count"].values.astype(np.int32).copy(),
                    }

            # Variables horaires locales
            day_data["hourly_local"] = {}
            for h in range(24):
                day_data["hourly_local"][h] = {}
                for p in ALL_PARAMS:
                    day_data["hourly_local"][h][p] = {
                        "sum":   ds[f"{safe}_h{h:02d}loc_{p}_sum"].values.copy(),
                        "sum2":  ds[f"{safe}_h{h:02d}loc_{p}_sum2"].values.copy(),
                        "count": ds[f"{safe}_h{h:02d}loc_{p}_count"].values.astype(np.int32).copy(),
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


def plot_cloud_height_distribution(grid, day=None, d1=None, d2=None,
                                    bins=50,
                                    xlim=(0, 5000),
                                    mask_type: str = "all",
                                    title=None):

    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)

    cloud_level = means["cloud_level_slwc"]
    surface_elevation = means["surface_elevation"]
    flag = means["land_water_flag"]

    if mask_type == "land":
        spatial_mask = np.round(flag) == 1
    elif mask_type == "ocean":
        spatial_mask = np.round(flag) == 0
    else:
        spatial_mask = np.ones(cloud_level.shape, dtype=bool)

    height = cloud_level - surface_elevation

    valid = spatial_mask & np.isfinite(height) & (height >= 0)
    height = height[valid].ravel()

    if len(height) == 0:
        raise ValueError("Aucune valeur valide.")

    counts, edges = np.histogram(height, bins=bins, range=xlim)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask_label = {"all": "Land and Ocean", "land": "Land",
                  "ocean": "Ocean"}.get(mask_type, mask_type)

    # Figure — height en Y, count en X
    fig, ax = plt.subplots(figsize=(6, 8))
    fig.patch.set_facecolor("white")

    ax.plot(counts, centers, 'r-', linewidth=2)

    ax.set_xlabel("Count", fontsize=11)
    ax.set_ylabel("Height agl (m)", fontsize=11)
    ax.set_ylim(xlim)
    ax.set_xlim(left=0)
    ax.tick_params(top=True, right=True, which="both", direction="in")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.set_title(title or f"SLW Cloud Height Above Ground Level\n"
                          f"{n_orbits} orbits | {mask_label} | {label}",
                 fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    plt.tight_layout()
    plt.show()

    return height

def plot_land_water_flag(grid, day=None, d1=None, d2=None):

    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)
    proj  = ccrs.SouthPolarStereo()
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])

    data = means["land_water_flag"]

    # Colormap discret : 0=ocean (bleu), 1=terre (vert)
    cmap  = mcolors.ListedColormap(["steelblue", "forestgreen"])
    norm  = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                 y_inline=True, alpha=0.3,
                 ylocs=np.arange(-90, -55, 10))

    pc = ax.pcolormesh(LON2D, LAT2D, np.round(data),
                       cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto")

    cbar = plt.colorbar(pc, ax=ax, orientation="horizontal",
                        shrink=0.8, pad=0.08,
                        ticks=[0.25, 0.75])
    cbar.set_ticklabels(["Ocean (0)", "Land (1)"])
    cbar.set_label("Land / Water flag", fontsize=10)

    ax.set_title(f"Land / Water Flag\n{n_orbits} orbits | grid {grid.dlat}°×{grid.dlon}° | {label}",
                 fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    plt.tight_layout()
    plt.show()

def plot_probability_density_trio(grid, param_x: str = "slwp", param_y: str = "temperature_slwc",
                                   day=None, d1=None, d2=None,
                                   bins_x=50, bins_y=50,
                                   xlim=(0, 20), ylim=(-40, 0),
                                   area_weighted=True, normalize=True,
                                   vmin=None, vmax=None, cmap="rainbow",
                                   log_density=False, mask_type: str = "all",
                                   elev_threshold_low: float = 1000,
                                   elev_threshold_high: float = 3000,
                                   # Seuils spécifiques land/ocean pour le mode "all"
                                   elev_land_low: float = 0,
                                   elev_land_high: float = 3800,
                                   elev_ocean_low: float = 0,
                                   elev_ocean_high: float = 5000,
                                   xlabel=None, ylabel=None):

    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    for p in (param_x, param_y):
        if p not in means:
            raise KeyError(f"Paramètre '{p}' absent. Disponibles : {list(means)}")

    data_x = means[param_x]
    data_y = means[param_y]
    elev   = means["surface_elevation"]
    flag   = means["land_water_flag"]

    lon_bins = np.asarray(grid.lon_bins)
    lat_bins = np.asarray(grid.lat_bins)
    lon_c = 0.5 * (lon_bins[:-1] + lon_bins[1:]) if len(lon_bins) == data_x.shape[1] + 1 else lon_bins
    lat_c = 0.5 * (lat_bins[:-1] + lat_bins[1:]) if len(lat_bins) == data_x.shape[0] + 1 else lat_bins
    LAT2D = np.meshgrid(lon_c, lat_c)[1]

    dlat = getattr(grid, "dlat", np.median(np.diff(lat_c)))
    dlon = getattr(grid, "dlon", np.median(np.diff(lon_c)))
    cell_area = np.cos(np.deg2rad(LAT2D)) * np.deg2rad(dlat) * np.deg2rad(dlon)

    # --- Masques de base ---
    land_mask  = flag >= 0.5   # dominant terre
    ocean_mask = flag <  0.5   # dominant océan

    if mask_type == "land":
        spatial_mask = land_mask
    elif mask_type == "ocean":
        spatial_mask = ocean_mask
    else:  # "all"
        spatial_mask = np.ones(data_x.shape, dtype=bool)

    base_valid = spatial_mask & np.isfinite(data_x) & np.isfinite(data_y)

    # --- Construction des 3 masques selon mask_type ---
    if mask_type == "all":
        # Panneau (a) : tout land + ocean avec leurs seuils respectifs
        mask_all = (
            base_valid & (
                (land_mask  & (elev >= elev_land_low)  & (elev < elev_land_high)) |
                (ocean_mask & (elev >= elev_ocean_low) & (elev < elev_ocean_high))
            )
        )
        # Panneau (b) : land < seuil haut land + ocean < seuil haut ocean
        mask_low = (
            base_valid & (
                (land_mask  & (elev < elev_land_low)) |
                (ocean_mask & (elev < elev_ocean_low))
            )
        )
        # Panneau (c) : land >= seuil haut + ocean >= seuil haut
        mask_high = (
            base_valid & (
                (land_mask  & (elev >= elev_land_high)) |
                (ocean_mask & (elev >= elev_ocean_high))
            )
        )
        masks = [
            (mask_all,
             f"All | land elev < {elev_land_high} m & ocean elev < {elev_ocean_high} m"),
            (mask_low,
             f"Land elev < {elev_land_low} m & Ocean elev < {elev_ocean_low} m"),
            (mask_high,
             f"Land elev ≥ {elev_land_high} m & Ocean elev ≥ {elev_ocean_high} m"),
        ]

    else:
        # Mode land ou ocean uniquement — seuils simples
        masks = [
            (base_valid,
             f"All elevations"),
            (base_valid & (elev < elev_threshold_low),
             f"Surface elevation < {elev_threshold_low} m"),
            (base_valid & (elev >= elev_threshold_high),
             f"Surface elevation ≥ {elev_threshold_high} m"),
        ]

    # --- Labels ---
    if ylabel is None:
        if "temperature_slwc" in param_y:
            ylabel = "In-Cloud SLW Temperature (°C)"
        elif "temperature" in param_y:
            ylabel = "Temperature (°C)"
        elif "elevation" in param_y:
            ylabel = "Elevation (m)"
        else:
            ylabel = param_y.replace("_", " ").capitalize()

    if xlabel is None:
        if "slwp" in param_x or "lwp" in param_x:
            xlabel = f"{param_x.upper()} ($g/m²$)"
        elif "lwc" in param_x:
            xlabel = f"{param_x.upper()} ($g/m³$)"
        else:
            xlabel = param_x.upper()

    mask_label = {"all": "Land and Ocean", "land": "Land",
                  "ocean": "Ocean"}.get(mask_type, mask_type)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True,
                              gridspec_kw={"wspace": 0.05})
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{param_x.upper()} vs In-Cloud SLW Temperature | {mask_label}\n"
                 f"{n_orbits} orbits | grid {dlat}°×{dlon}° | {label}",
                 fontsize=13, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, (valid, panel_title), plabel in zip(axes, masks, panel_labels):

        x = data_x[valid].ravel()
        y = data_y[valid].ravel()
        w = cell_area[valid].ravel() if area_weighted else None

        if x.size == 0:
            ax.set_title(f"{plabel} {panel_title}\nNo data", fontsize=10)
            continue

        print(f"{plabel} {panel_title} → N={x.size}")

        # Histogramme 2D
        y_range    = ylim if ylim is not None else [y.min(), y.max()]
        hist_range = [xlim, y_range]
        H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y],
                                            range=hist_range, weights=w)
        if H.sum() > 0:
            H = H / H.sum() * 100
        H_plot = np.where(H.T > 0, H.T, np.nan)

        cmap_obj = plt.get_cmap(cmap)
        lo = vmin if vmin is not None else 0
        hi = vmax if vmax is not None else np.nanmax(H_plot)

        if log_density:
            norm = mcolors.LogNorm(vmin=vmin or 1e-3, vmax=vmax or hi)
        else:
            norm = mcolors.Normalize(vmin=lo, vmax=hi)

        pc = ax.pcolormesh(xedges, yedges, H_plot, cmap=cmap_obj,
                           norm=norm, shading="auto")
        cbar = plt.colorbar(pc, ax=ax, orientation="horizontal",
                            fraction=0.05, pad=0.08)
        cbar.set_label("Probability density (%)", fontsize=9)

        ax.set_xlabel(xlabel, fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel(ylabel, fontsize=11)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(top=True, right=True, which="both", direction="in")
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.set_title(f"{plabel} {panel_title}", fontsize=11,
                     fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

        _add_ricaud_curve(ax)

    plt.tight_layout()
    plt.show()

def plot_quality_flag(orbites: list[dict]) -> None:

    counts = Counter(str(orb.get("quality_flag", "UNKNOWN")).strip().upper() for orb in orbites)

    # Print les orbites NG
    ng_orbites = [orb.get("nom_orbite", orb.get("orbit_id", "unknown"))
                  for orb in orbites
                  if str(orb.get("quality_flag", "")).strip().upper() == "NG"]
    if ng_orbites:
        print(f"\n{len(ng_orbites)} orbite(s) NG :")
        for nom in ng_orbites:
            print(f"  → {nom}")
    else:
        print("Aucune orbite NG.")

    labels = [f for f in FLAG_ORDER if counts.get(f, 0) > 0]
    extras = sorted(k for k in counts if k not in FLAG_ORDER)
    labels += extras

    values = [counts[l] for l in labels]
    colors = [FLAG_COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.5)

    total = sum(values)
    for bar, val in zip(bars, values):
        pct = 100.0 * val / total if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val}\n({pct:.1f} %)", ha="center", va="bottom",
                fontsize=10, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)

    ax.set_ylabel("Number of orbits", fontsize=11, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_xlabel("Quality flag", fontsize=11, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_title("Orbit quality flag", fontsize=13, color=TITLE_COLOR, fontweight=TITLE_WEIGHT)
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(True)
    ax.tick_params(axis="both", labelsize=11)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()