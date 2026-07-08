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