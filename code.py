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