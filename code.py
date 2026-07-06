def plot_slwp_vs_temperature_full(orbites: list,
                                   xlim=(0, 15),
                                   ylim=(-40, -10),
                                   bins_x=50, bins_y=50,
                                   bins_lwp=None,
                                   normalize=True,
                                   cmap="rainbow"):

    all_slwp = []
    all_temp = []

    for orbit_data in orbites:
        height = orbit_data["height"]
        lwc    = orbit_data["lwc_plot"]
        lwp    = orbit_data["lwp_plot"]
        temp   = orbit_data["temp_c"]

        idx_valid = np.where(np.asarray(lwp) > 0)[0]

        for idx in idx_valid:
            i = int(idx)

            altitude_ref = height[i, ::-1]
            lwc_profile  = lwc[i, ::-1]
            temp_profile = temp[i, ::-1]

            lwc_profile  = np.ma.masked_where((lwc_profile < 0) | (lwc_profile < -9000), lwc_profile)
            temp_profile = np.ma.masked_where(temp_profile < -200, temp_profile)

            slwc_profile = np.where(temp_profile <= 0, lwc_profile, np.nan)
            slwc_profile = np.ma.masked_where(slwc_profile < -900, slwc_profile)

            valid        = ~np.isnan(altitude_ref)
            altitude_ref = altitude_ref[valid]
            lwc_profile  = lwc_profile[valid]
            temp_profile = temp_profile[valid]
            slwc_profile = slwc_profile[valid]

            slwp_i = float(np.nansum(np.ma.filled(slwc_profile, 0.0)) * 100)
            if slwp_i <= 0:
                continue

            slwc_values = np.ma.filled(slwc_profile, 0.0)
            mask_slwc   = slwc_values > 0
            if np.sum(mask_slwc) == 0:
                continue

            temp_mean_i = float(np.nanmean(np.ma.filled(temp_profile, np.nan)[mask_slwc]))
            if np.isnan(temp_mean_i):
                continue

            all_slwp.append(slwp_i)
            all_temp.append(temp_mean_i)

    x = np.array(all_slwp)
    y = np.array(all_temp)

    print(f"Total pixels valides : {len(x)}")
    print(f"SLWP min/max : {x.min():.4f} / {x.max():.4f} g/m²")
    print(f"Temp min/max : {y.min():.2f} / {y.max():.2f} °C")

    # Régression logarithmique
    from scipy.optimize import curve_fit

    def log_func(x, a, b):
        return a * np.log(x) + b

    mask_pos = x > 0
    try:
        popt, pcov = curve_fit(log_func, x[mask_pos], y[mask_pos])
        a, b       = popt
        perr       = np.sqrt(np.diag(pcov))
        x_line     = np.linspace(x[mask_pos].min(), x[mask_pos].max(), 100)
        y_line     = log_func(x_line, a, b)
        y_line_up  = log_func(x_line, a + perr[0], b + perr[1])
        y_line_dn  = log_func(x_line, a - perr[0], b - perr[1])
        y_pred     = log_func(x[mask_pos], a, b)
        ss_res     = np.sum((y[mask_pos] - y_pred) ** 2)
        ss_tot     = np.sum((y[mask_pos] - np.mean(y[mask_pos])) ** 2)
        r2         = 1 - ss_res / ss_tot
        reg_label  = f'T = {a:.2f}·ln(SLWP) + {b:.2f}\nR²={r2:.3f}'
    except Exception as e:
        print(f"Régression échouée : {e}")
        x_line = y_line = y_line_up = y_line_dn = None
        reg_label = ""

    # Bins de LWP pour le panneau (b)
    if bins_lwp is None:
        bins_lwp = np.arange(xlim[0], xlim[1] + 1, 1)

    bin_centers  = 0.5 * (bins_lwp[:-1] + bins_lwp[1:])
    bin_means    = []
    bin_stds     = []
    bin_centers_valid = []

    for k in range(len(bins_lwp) - 1):
        mask_bin = (x >= bins_lwp[k]) & (x < bins_lwp[k + 1])
        if np.sum(mask_bin) >= 5:  # au moins 5 points
            bin_means.append(np.mean(y[mask_bin]))
            bin_stds.append(np.std(y[mask_bin]))
            bin_centers_valid.append(bin_centers[k])

    bin_centers_valid = np.array(bin_centers_valid)
    bin_means         = np.array(bin_means)
    bin_stds          = np.array(bin_stds)

    # Figure 3 panneaux
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7),
                                         sharey=True,
                                         gridspec_kw={"wspace": 0.05})

    # --- Panneau (a) : densité de probabilité ---
    y_range  = ylim
    H, xedges, yedges = np.histogram2d(x, y,
                                        bins=[bins_x, bins_y],
                                        range=[xlim, y_range])
    if normalize and H.max() > 0:
        H = H / H.max()
    H_plot = np.where(H.T > 0, H.T, np.nan)

    cmap_obj = plt.get_cmap(cmap)
    norm     = mcolors.LogNorm(vmin=0.01, vmax=H_plot[~np.isnan(H_plot)].max())
    pc       = ax1.pcolormesh(xedges, yedges, H_plot, cmap=cmap_obj,
                               norm=norm, shading="auto")
    cbar = fig.colorbar(pc, ax=ax1, orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("PD (%)", fontsize=9)
    ax1.set_xlabel("SLWP (g/m²)", fontsize=11)
    ax1.set_ylabel("Température (°C)", fontsize=11)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title("(a) Densité de probabilité", fontsize=11)
    ax1.axhline(-40, color="grey", linewidth=0.8, linestyle="--")
    ax1.axhline(0,   color="k",    linewidth=0.8, linestyle="--")

    # --- Panneau (b) : moyenne par bin + régression log ---
    ax2.errorbar(bin_means, bin_centers_valid,
                 xerr=bin_stds,
                 fmt='r*', markersize=8, linewidth=1.2,
                 ecolor='red', capsize=3, label='Moyenne ± std')

    if x_line is not None:
        ax2.plot(y_line,    x_line, 'b-',  linewidth=2,   label=reg_label)
        ax2.plot(y_line_up, x_line, 'b--', linewidth=1, alpha=0.7)
        ax2.plot(y_line_dn, x_line, 'b--', linewidth=1, alpha=0.7)

    ax2.set_xlabel("SLWP (g/m²)", fontsize=11)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title("(b) Moyenne par bin + régression log", fontsize=11)
    ax2.axhline(-40, color="grey", linewidth=0.8, linestyle="--")
    ax2.axhline(0,   color="k",    linewidth=0.8, linestyle="--")
    ax2.legend(fontsize=8)

    # --- Panneau (c) : histogramme 1D de la température ---
    temp_bins   = np.linspace(ylim[0], ylim[1], bins_y + 1)
    temp_counts, temp_edges = np.histogram(y, bins=temp_bins)

    ax3.plot(temp_counts / 1000, 0.5 * (temp_edges[:-1] + temp_edges[1:]),
             'r-', linewidth=2)
    ax3.set_xlabel("Count Number x 10³", fontsize=11)
    ax3.set_xlim(left=0)
    ax3.set_ylim(ylim)
    ax3.set_title("(c) Distribution température SLW", fontsize=11)
    ax3.axhline(-40, color="grey", linewidth=0.8, linestyle="--")
    ax3.axhline(0,   color="k",    linewidth=0.8, linestyle="--")

    fig.suptitle(f"SLWP vs Température SLW — {len(orbites)} orbites",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("slwp_vs_temperature_full.png", dpi=150)
    plt.show()


    # Masque continental (calculé une seule fois avant la boucle)
def plot_temperature_slwc_histogram(orbites: list,
                                     xlim=(-50, 0),
                                     bins=50,
                                     land_only=True,
                                     land_resolution="50m",
                                     title=None):

    # Masque continental (calculé une seule fois avant la boucle)
    if land_only:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        from shapely import contains_xy

        land_shp  = shpreader.natural_earth(resolution=land_resolution,
                                             category="physical", name="land")
        land_geom = unary_union(list(shpreader.Reader(land_shp).geometries()))

    all_temp = []

    for orbit_data in orbites:
        height = orbit_data["height"]
        lwc    = orbit_data["lwc_plot"]
        lwp    = orbit_data["lwp_plot"]
        temp   = orbit_data["temp_c"]
        lat    = np.asarray(orbit_data["lat"], dtype=float)
        lon    = np.asarray(orbit_data["lon"], dtype=float)

        # Masque continental pour cette orbite
        if land_only:
            is_land   = contains_xy(land_geom, lon, lat)
            idx_valid = np.where((np.asarray(lwp) > 0) & is_land)[0]
        else:
            idx_valid = np.where(np.asarray(lwp) > 0)[0]

        for idx in idx_valid:
            i = int(idx)

            altitude_ref = height[i, ::-1]
            lwc_profile  = lwc[i, ::-1]
            temp_profile = temp[i, ::-1]

            lwc_profile  = np.ma.masked_where((lwc_profile < 0) | (lwc_profile < -9000), lwc_profile)
            temp_profile = np.ma.masked_where(temp_profile < -200, temp_profile)

            slwc_profile = np.where(temp_profile <= 0, lwc_profile, np.nan)
            slwc_profile = np.ma.masked_where(slwc_profile < -900, slwc_profile)

            valid        = ~np.isnan(altitude_ref)
            altitude_ref = altitude_ref[valid]
            temp_profile = temp_profile[valid]
            slwc_profile = slwc_profile[valid]

            slwp_i = float(np.nansum(np.ma.filled(slwc_profile, 0.0)) * 100)
            if slwp_i <= 0:
                continue

            slwc_values = np.ma.filled(slwc_profile, 0.0)
            mask_slwc   = slwc_values > 0
            if np.sum(mask_slwc) == 0:
                continue

            temp_filled = np.ma.filled(temp_profile, np.nan)
            temp_mean_i = float(np.nanmean(temp_filled[mask_slwc]))

            if np.isnan(temp_mean_i):
                continue

            all_temp.append(temp_mean_i)

    y = np.array(all_temp)
    print(f"Total pixels SLW valides : {len(y)}")
    print(f"Temp min/max : {y.min():.2f} / {y.max():.2f} °C")
    print(f"Temp moyenne : {np.mean(y):.2f} °C")
    print(f"Temp médiane : {np.median(y):.2f} °C")

    if len(y) == 0:
        raise ValueError("Aucun pixel valide trouvé.")

    # Histogramme brut
    counts, edges = np.histogram(y, bins=bins, range=xlim)
    centers       = 0.5 * (edges[:-1] + edges[1:])

    # Figure — température en Y, count en X
    fig, ax = plt.subplots(figsize=(6, 8))

    ax.plot(counts, centers, 'r-', linewidth=2)

    ax.tick_params(top=True, right=True, which="both", direction="in")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.set_ylabel("Mean temperature (°C)", fontsize=11)
    ax.set_xlabel("Count Number", fontsize=11)
    ax.set_ylim(xlim)
    ax.set_xlim(left=0)

    land_label = "continent uniquement" if land_only else "continent + océan"
    ax.set_title(title or f"Distribution de température des nuages SLW\n"
                          f"{len(orbites)} orbites | N={len(y)} pixels",
                 fontsize=11)

    plt.tight_layout()
    plt.show()

    return y