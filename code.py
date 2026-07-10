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