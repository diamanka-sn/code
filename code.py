def plot_probability_density_trio(grid, param_x: str = "slwp", param_y: str = "temperature_slwc",
                                   day=None, d1=None, d2=None,
                                   bins_x=50, bins_y=50,
                                   xlim=(0, 20), ylim=(-45, 0),
                                   area_weighted=True, normalize=True,
                                   vmin=None, vmax=None, cmap="rainbow",
                                   log_density=False,
                                   mask_type: str = "all",   # "all", "land", "ocean"
                                   elev_threshold: float = 2000,  # seuil en mètres
                                   xlabel=None, ylabel=None):
    """
    Trace 3 PD côte à côte :
        (a) tout (pas de filtre élévation)
        (b) surface_elevation < elev_threshold
        (c) surface_elevation >= elev_threshold
    """

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

    # --- Masque spatial ---
    if mask_type == "land":
        spatial_mask = np.round(flag) == 1
    elif mask_type == "ocean":
        spatial_mask = np.round(flag) == 0
    else:
        spatial_mask = np.ones(data_x.shape, dtype=bool)

    base_valid = spatial_mask & np.isfinite(data_x) & np.isfinite(data_y)

    # --- 3 masques d'élévation ---
    masks = [
        (base_valid,                                          f"All ({mask_type})"),
        (base_valid & (elev <  elev_threshold),              f"Elevation < {elev_threshold} m"),
        (base_valid & (elev >= elev_threshold),              f"Elevation ≥ {elev_threshold} m"),
    ]

    # --- Labels ---
    if ylabel is None:
        if "temperature" in param_y:
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

    mask_label = {"all": "Land + Ocean", "land": "Land only",
                  "ocean": "Ocean only"}.get(mask_type, mask_type)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True,
                              gridspec_kw={"wspace": 0.05})
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{param_x.upper()} vs {param_y.replace('_', ' ')} | {mask_label}\n"
                 f"{n_orbits} orbits | grid {dlat}°×{dlon}° | {label}",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, (valid, panel_title), plabel in zip(axes, masks, panel_labels):

        x = data_x[valid].ravel()
        y = data_y[valid].ravel()
        w = cell_area[valid].ravel() if area_weighted else None

        if x.size == 0:
            ax.set_title(f"{plabel} {panel_title}\nNo data", fontsize=10)
            continue

        print(f"{plabel} {panel_title} → N={x.size} | "
              f"{param_x} [{x.min():.2f}, {x.max():.2f}] | "
              f"{param_y} [{y.min():.2f}, {y.max():.2f}]")

        # Histogramme 2D
        y_range    = ylim if ylim is not None else [y.min(), y.max()]
        hist_range = [xlim, y_range]
        H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y],
                                            range=hist_range, weights=w)
        if normalize and H.sum() > 0:
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
        cbar = plt.colorbar(pc, ax=ax, orientation="vertical", fraction=0.05, pad=0.02)
        cbar.set_label("Probability density (%)" if normalize else "Density", fontsize=9)

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

        ax.set_title(f"{plabel} {panel_title}\nN={x.size}", fontsize=10,
                     fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

        _add_ricaud_curve(ax)

    plt.tight_layout()
    plt.show()


def plot_probability_density_trio(grid, param_x: str = "slwp", param_y: str = "temperature_slwc",
                                   day=None, d1=None, d2=None,
                                   bins_x=50, bins_y=50,
                                   xlim=(0, 20), ylim=(-45, 0),
                                   area_weighted=True, normalize=True,
                                   vmin=None, vmax=None, cmap="rainbow",
                                   log_density=False,
                                   mask_type: str = "all",   # "all", "land", "ocean"
                                   elev_threshold: float = 2000,  # seuil en mètres
                                   xlabel=None, ylabel=None):
    """
    Trace 3 PD côte à côte :
        (a) tout (pas de filtre élévation)
        (b) surface_elevation < elev_threshold
        (c) surface_elevation >= elev_threshold
    """

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

    # --- Masque spatial ---
    if mask_type == "land":
        spatial_mask = np.round(flag) == 1
    elif mask_type == "ocean":
        spatial_mask = np.round(flag) == 0
    else:
        spatial_mask = np.ones(data_x.shape, dtype=bool)

    base_valid = spatial_mask & np.isfinite(data_x) & np.isfinite(data_y)

    # --- 3 masques d'élévation ---
    masks = [
        (base_valid,                                          f"All ({mask_type})"),
        (base_valid & (elev <  elev_threshold),              f"Elevation < {elev_threshold} m"),
        (base_valid & (elev >= elev_threshold),              f"Elevation ≥ {elev_threshold} m"),
    ]

    # --- Labels ---
    if ylabel is None:
        if "temperature" in param_y:
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

    mask_label = {"all": "Land + Ocean", "land": "Land only",
                  "ocean": "Ocean only"}.get(mask_type, mask_type)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True,
                              gridspec_kw={"wspace": 0.05})
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{param_x.upper()} vs {param_y.replace('_', ' ')} | {mask_label}\n"
                 f"{n_orbits} orbits | grid {dlat}°×{dlon}° | {label}",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, (valid, panel_title), plabel in zip(axes, masks, panel_labels):

        x = data_x[valid].ravel()
        y = data_y[valid].ravel()
        w = cell_area[valid].ravel() if area_weighted else None

        if x.size == 0:
            ax.set_title(f"{plabel} {panel_title}\nNo data", fontsize=10)
            continue

        print(f"{plabel} {panel_title} → N={x.size} | "
              f"{param_x} [{x.min():.2f}, {x.max():.2f}] | "
              f"{param_y} [{y.min():.2f}, {y.max():.2f}]")

        # Histogramme 2D
        y_range    = ylim if ylim is not None else [y.min(), y.max()]
        hist_range = [xlim, y_range]
        H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y],
                                            range=hist_range, weights=w)
        if normalize and H.sum() > 0:
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
        cbar = plt.colorbar(pc, ax=ax, orientation="vertical", fraction=0.05, pad=0.02)
        cbar.set_label("Probability density (%)" if normalize else "Density", fontsize=9)

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

        ax.set_title(f"{plabel} {panel_title}\nN={x.size}", fontsize=10,
                     fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

        _add_ricaud_curve(ax)

    plt.tight_layout()
    plt.show() 


    if ylabel is None:
    if "temperature_slwc" in param_y:
        ylabel = "In-Cloud SLW Temperature (°C)"
    elif "temperature" in param_y:
        ylabel = "Temperature (°C)"
    elif "elevation" in param_y:
        ylabel = "Elevation (m)"
    else:
        ylabel = param_y.replace("_", " ").capitalize()

        # Tout
# Terre uniquement, seuil à 2000 m
plot_probability_density_trio(grid, mask_type="land", elev_threshold=2000,
                               d1="2025-12-01", d2="2025-12-31")

# Océan uniquement, seuil à 0 m
plot_probability_density_trio(grid, mask_type="ocean", elev_threshold=0,
                               d1="2025-12-01", d2="2025-12-31")

# Tout, seuil à 1000 m
plot_probability_density_trio(grid, mask_type="all", elev_threshold=1000,
                               d1="2025-12-01", d2="2025-12-31")