def plot_slwc_maps(grid, day=None, d1=None, d2=None,
                   vmin_temp=-40, vmax_temp=0,
                   vmin_alt=0,   vmax_alt=8000,
                   vmin_slwp=0,  vmax_slwp=20):

    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)
    proj  = ccrs.SouthPolarStereo()
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])

    fig = plt.figure(figsize=(24, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Nuages SLW | {n_orbits} orbites | grille {grid.dlat}°×{grid.dlon}°\n{label}",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    panels = [
        (means["slwp"],             "SLWP (g/m²)",         vmin_slwp, vmax_slwp, "rainbow",  "SLWP moyen"),
        (means["temperature_slwc"], "Température SLW (°C)", vmin_temp, vmax_temp, "coolwarm", "Température moyenne SLW"),
        (means["cloud_level_slwc"], "Altitude (m)",         vmin_alt,  vmax_alt,  "rainbow",  "Altitude moyenne nuages SLW"),
    ]

    for col, (data, cbar_label, vmin, vmax, cmap, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, col, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)

        pc = ax.pcolormesh(LON2D, LAT2D, data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        plt.colorbar(pc, ax=ax, orientation="horizontal",
                     shrink=0.8, pad=0.04, label=cbar_label)

        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".",
                markersize=5, linestyle="none",
                transform=ccrs.PlateCarree(), label="Dome C")
        _add_distance_circles(ax)
        ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
        ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_slwc_maps(grid, day=None, d1=None, d2=None,
                   vmin_temp=-40, vmax_temp=0,
                   vmin_alt=0,   vmax_alt=8000,
                   vmin_slwp=0,  vmax_slwp=20):

    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)
    proj  = ccrs.SouthPolarStereo()
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])

    fig = plt.figure(figsize=(24, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"SLW Clouds | {n_orbits} orbits | grid {grid.dlat}°×{grid.dlon}°\n{label}",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    panels = [
        (means["slwp"],             "SLWP (g/m²)",            vmin_slwp, vmax_slwp, "rainbow",  "Mean SLWP"),
        (means["temperature_slwc"], "SLW Temperature (°C)",   vmin_temp, vmax_temp, "coolwarm", "Mean SLW Temperature"),
        (means["cloud_level_slwc"], "Altitude (m)",            vmin_alt,  vmax_alt,  "rainbow",  "Mean SLW Cloud Altitude"),
    ]

    for col, (data, cbar_label, vmin, vmax, cmap, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, col, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)

        pc = ax.pcolormesh(LON2D, LAT2D, data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        plt.colorbar(pc, ax=ax, orientation="horizontal",
                     shrink=0.8, pad=0.04, label=cbar_label)

        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".",
                markersize=5, linestyle="none",
                transform=ccrs.PlateCarree(), label="Dome C")
        _add_distance_circles(ax)
        ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
        ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_slwc_maps_std(grid, day=None, d1=None, d2=None,
                       vmin_temp=0, vmax_temp=10,
                       vmin_alt=0,  vmax_alt=2000,
                       vmin_slwp=0, vmax_slwp=5):

    _, stds, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)
    proj  = ccrs.SouthPolarStereo()
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])

    fig = plt.figure(figsize=(24, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"SLW Clouds — Standard Deviation | {n_orbits} orbits | grid {grid.dlat}°×{grid.dlon}°\n{label}",
                 fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    panels = [
        (stds["slwp"], "SLWP std (g/m²)", vmin_slwp, vmax_slwp, "rainbow", "Std SLWP"),
        (stds["temperature_slwc"], "SLW Temperature std (°C)",  vmin_temp, vmax_temp, "coolwarm", "Std SLW Temperature"),
        (stds["cloud_level_slwc"], "Altitude std (m)", vmin_alt,  vmax_alt, "rainbow",  "Std SLW Cloud Altitude"),
    ]

    for col, (data, cbar_label, vmin, vmax, cmap, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, col, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)

        pc = ax.pcolormesh(LON2D, LAT2D, data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        plt.colorbar(pc, ax=ax, orientation="horizontal",
                     shrink=0.8, pad=0.04, label=cbar_label)

        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".",
                markersize=5, linestyle="none",
                transform=ccrs.PlateCarree(), label="Dome C")
        _add_distance_circles(ax)
        ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
        ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_slwc_maps(grid, day=None, d1=None, d2=None,
                   vmin_temp=-40, vmax_temp=0,
                   vmin_alt=0,   vmax_alt=8000,
                   vmin_slwp=0,  vmax_slwp=20,
                   vmin_height=0, vmax_height=5000):

    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    LON2D, LAT2D = np.meshgrid(grid.lon_bins, grid.lat_bins)
    proj  = ccrs.SouthPolarStereo()
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])

    # Hauteur au-dessus de la surface = altitude - élévation de la surface
    height_above_surface = means["cloud_level_slwc"] - means["surface_elevation"]

    # --- Figure 1 : Moyenne (SLWP, Température, Altitude) ---
    fig1 = plt.figure(figsize=(24, 8))
    fig1.patch.set_facecolor("white")
    fig1.suptitle(f"SLW Clouds — Mean | {n_orbits} orbits | grid {grid.dlat}°×{grid.dlon}°\n{label}",
                  fontsize=12, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    panels_mean = [
        (means["slwp"],             "SLWP (g/m²)",          vmin_slwp, vmax_slwp, "rainbow",  "Mean SLWP"),
        (means["cloud_level_slwc"], "Altitude (m)",          vmin_alt,  vmax_alt,  "rainbow",  "Mean SLW Cloud Altitude"),
        (height_above_surface,      "Height above surface (m)", vmin_height, vmax_height, "rainbow", "Mean SLW Cloud Height above Surface"),
    ]

    for col, (data, cbar_label, vmin, vmax, cmap, title) in enumerate(panels_mean, 1):
        ax = fig1.add_subplot(1, 3, col, projection=proj)
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.3)

        pc = ax.pcolormesh(LON2D, LAT2D, data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        plt.colorbar(pc, ax=ax, orientation="horizontal",
                     shrink=0.8, pad=0.04, label=cbar_label)

        ax.plot(LON_REF, LAT_REF, color="#FFA500", marker=".",
                markersize=5, linestyle="none",
                transform=ccrs.PlateCarree(), label="Dome C")
        _add_distance_circles(ax)
        ax.set_title(title, fontsize=10, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)
        ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_probability_density(grid, param_x: str = "slwp", param_y: str = "temperature_slwc",
                              day=None, d1=None, d2=None,
                              bins_x=50, bins_y=50,
                              xlim=(0, 20), ylim=(-45, 0),
                              area_weighted=True, normalize=True,
                              boundary_norm=False,
                              bounds=None, xlabel=None, ylabel=None,
                              vmin=None, vmax=None, cmap="rainbow",
                              log_density=False,
                              # Filtre spatial
                              mask_type: str = "all",        # "all", "land", "ocean"
                              # Filtre sur altitude du nuage SLW
                              cloud_alt_min: float | None = None,
                              cloud_alt_max: float | None = None,
                              ):
    """
    mask_type :
        "all"   → tout (terre + océan)
        "land"  → terre uniquement  (land_water_flag == 1)
        "ocean" → océan uniquement  (land_water_flag == 0)

    cloud_alt_min / cloud_alt_max :
        filtre sur cloud_level_slwc (altitude moyenne des nuages SLW en mètres).
        Exemples :
            cloud_alt_min=3000              → nuages SLW au-dessus de 3000 m
            cloud_alt_max=2000              → nuages SLW en dessous de 2000 m
            cloud_alt_min=1000, cloud_alt_max=4000 → nuages entre 1000 et 4000 m
    """
    means, _, n_orbits, label = _resolve_grid_range(grid, day, d1, d2)
    for p in (param_x, param_y):
        if p not in means:
            raise KeyError(f"Paramètre '{p}' absent. Disponibles : {list(means)}")

    data_x     = means[param_x]
    data_y     = means[param_y]
    cloud_alt  = means["cloud_level_slwc"]
    flag       = means["land_water_flag"]

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
    else:  # "all"
        spatial_mask = np.ones(data_x.shape, dtype=bool)

    # --- Filtre sur l'altitude du nuage SLW ---
    alt_mask = np.ones(data_x.shape, dtype=bool)
    if cloud_alt_min is not None:
        alt_mask &= cloud_alt >= cloud_alt_min
    if cloud_alt_max is not None:
        alt_mask &= cloud_alt <= cloud_alt_max

    # --- Masque final ---
    valid = spatial_mask & alt_mask & np.isfinite(data_x) & np.isfinite(data_y)
    x = data_x[valid].ravel()
    y = data_y[valid].ravel()
    w = cell_area[valid].ravel() if area_weighted else None

    if x.size == 0:
        raise ValueError("Aucun point valide après masquage.")

    print(f"N points valides : {x.size}")
    print(f"{param_x} min/max : {x.min():.4f} / {x.max():.4f}")
    print(f"{param_y} min/max : {y.min():.4f} / {y.max():.4f}")

    # --- Histogramme 2D ---
    y_range    = ylim if ylim is not None else [y.min(), y.max()]
    hist_range = [xlim, y_range] if xlim is not None else [None, y_range]
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y],
                                        range=hist_range, weights=w)
    if normalize and H.sum() > 0:
        H = H / H.sum() * 100
    H_plot = np.where(H.T > 0, H.T, np.nan)

    # --- Figure ---
    cmap_obj = plt.get_cmap(cmap)
    lo = vmin if vmin is not None else 0
    hi = vmax if vmax is not None else np.nanmax(H_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    if log_density:
        norm = mcolors.LogNorm(vmin=vmin or 1e-3, vmax=vmax or hi)
        pc   = ax.pcolormesh(xedges, yedges, H_plot, cmap=cmap_obj, norm=norm, shading="auto")
        cbar = plt.colorbar(pc, ax=ax)
    elif boundary_norm:
        b    = bounds if bounds is not None else np.linspace(lo, hi, 11)
        norm = mcolors.BoundaryNorm(boundaries=b, ncolors=cmap_obj.N, extend="max")
        pc   = ax.pcolormesh(xedges, yedges, H_plot, cmap=cmap_obj, norm=norm, shading="auto")
        cbar = plt.colorbar(pc, ax=ax)
        cbar.set_ticks(b)
    else:
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        pc   = ax.pcolormesh(xedges, yedges, H_plot, cmap=cmap_obj, norm=norm, shading="auto")
        cbar = plt.colorbar(pc, ax=ax)

    cbar.set_label("Probability density (%)" if normalize else "Density (area-weighted)",
                   fontsize=10)

    # --- Labels automatiques ---
    if ylabel is None:
        if "temperature" in param_y:
            ylabel = "Temperature (°C)"
        elif "elevation" in param_y:
            ylabel = "Elevation (m)"
        elif "cloud_level" in param_y:
            ylabel = "Cloud altitude (m)"
        else:
            ylabel = param_y.replace("_", " ").capitalize()

    if xlabel is None:
        if "slwp" in param_x or "lwp" in param_x:
            xlabel = f"{param_x.upper()} ($g/m²$)"
        elif "lwc" in param_x:
            xlabel = f"{param_x.upper()} ($g/m³$)"
        else:
            xlabel = param_x.upper()

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(top=True, right=True, which="both", direction="in")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)

    # --- Titre avec info masque ---
    mask_label = {"all": "Land + Ocean", "land": "Land only",
                  "ocean": "Ocean only"}.get(mask_type, mask_type)
    alt_label = ""
    if cloud_alt_min is not None and cloud_alt_max is not None:
        alt_label = f" | {cloud_alt_min} m ≤ cloud alt ≤ {cloud_alt_max} m"
    elif cloud_alt_min is not None:
        alt_label = f" | cloud alt ≥ {cloud_alt_min} m"
    elif cloud_alt_max is not None:
        alt_label = f" | cloud alt ≤ {cloud_alt_max} m"

    ax.set_title(f"{param_x.upper()} vs {param_y.replace('_', ' ')}\n"
                 f"{n_orbits} orbits | grid {dlat}°×{dlon}° | {label}\n"
                 f"{mask_label}{alt_label}",
                 fontsize=11, fontweight=TITLE_WEIGHT, color=TITLE_COLOR)

    _add_ricaud_curve(ax)
    plt.tight_layout()
    plt.show()


# Tout
plot_probability_density(grid, mask_type="all", d1="2025-12-01", d2="2025-12-31")

# Terre, nuages SLW au-dessus de 3000 m
plot_probability_density(grid, mask_type="land", cloud_alt_min=3000, d1="2025-12-01", d2="2025-12-31")

# Océan, nuages SLW en dessous de 2000 m
plot_probability_density(grid, mask_type="ocean", cloud_alt_max=2000, d1="2025-12-01", d2="2025-12-31")

# Terre, nuages entre 1000 et 4000 m
plot_probability_density(grid, mask_type="land", cloud_alt_min=1000, cloud_alt_max=4000, d1="2025-12-01", d2="2025-12-31")