def fit_subplots_without_lensing(
    fit,
    transformer,
    grid=None,
    centre=None,
    radius=None,
    normalize_residuals=True,
    cmap="jet",
    show_contours=True,
    xlim_image_plane=None,
    ylim_image_plane=None,
    xlim_source_plane=None,
    ylim_source_plane=None,
    show_critical_curves=True,
    show_caustic_curves=True,
    centre_from_mask=(0.0, 0.0),
    apply_mask=True,
    save=False,
    return_axes=False,
    output_filename="fit_subplots.png",
    show=True,
):
    def normalize(array, min_value=-1.0, max_value=+1.0):
        return min_value + (max_value - min_value) * (
            (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
        )

    # NOTE: This must be the "initial" grid, not the masked grid (this is because
    # the masked grid has a different extent than the "initial" grid)
    if grid is None:
        grid = fit.masked_dataset.grid

    # real_space_mask = fit.masked_dataset.real_space_mask[::-1, :]

    extent = autoarray_mask_utils.extent_from_mask(
        mask=fit.masked_dataset.real_space_mask
    )

    dirty_image, dirty_model_image = [
        transformer.image_from_visibilities(visibilities=visibilities)
        for visibilities in [fit.data, fit.model_data]
    ]

    vmin = np.nanmin(dirty_image)
    vmax = np.nanmax(dirty_image)

    # NOTE:

    figure, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    axes[0].imshow(
        dirty_image,
        origin="lower",
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    axes[1].imshow(
        dirty_model_image,
        origin="lower",
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    residuals = np.subtract(dirty_image, dirty_model_image)
    if normalize_residuals:
        axes[2].imshow(
            normalize(residuals),
            origin="lower",
            cmap=cmap,
            extent=extent,
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
        )
    else:
        axes[2].imshow(
            residuals,
            origin="lower",
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

    model_image = autoarray_arrays_class_attributes.in_2d(
        a=autolens_tracer_class_attributes.image_from_grid(tracer=fit.tracer, grid=grid)
    )[::-1, :]

    axes[3].imshow(model_image, origin="lower", cmap=cmap, extent=extent, aspect="auto")

    for i in [1, 2, 3]:
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    if xlim_image_plane is not None and ylim_image_plane is not None:
        ylim_image_plane_shifted = (
            ylim_image_plane[0] + centre_from_mask[0],
            ylim_image_plane[1] + centre_from_mask[0],
        )
        xlim_image_plane_shifted = (
            xlim_image_plane[0] + centre_from_mask[1],
            xlim_image_plane[1] + centre_from_mask[1],
        )

        xticks_image_plane = np.linspace(
            xlim_image_plane_shifted[0], xlim_image_plane_shifted[1], 5
        )
        yticks_image_plane = np.linspace(
            ylim_image_plane_shifted[0], ylim_image_plane_shifted[1], 5
        )

        xticks_labels_image_plane = np.linspace(
            xlim_image_plane[0], xlim_image_plane[1], 5
        )
        yticks_labels_image_plane = np.linspace(
            ylim_image_plane[0], ylim_image_plane[1], 5
        )

        for i in [0, 1, 2, 3]:
            axes[i].set_xlim(xlim_image_plane_shifted)
            axes[i].set_ylim(ylim_image_plane_shifted)

        axes[2].text(
            xlim_image_plane_shifted[0] + 0.05 * abs(xlim_image_plane_shifted[0]),
            ylim_image_plane_shifted[0] + 0.10 * abs(ylim_image_plane_shifted[0]),
            r"$\chi^2 = {:.2f}$".format(fit.chi_squared),
            fontsize=15,
        )

        axes[0].set_xticks(xticks_image_plane)
        axes[0].set_yticks(yticks_image_plane)
        axes[0].set_xticklabels(xticks_labels_image_plane)
        axes[0].set_yticklabels(yticks_labels_image_plane)

    axes[0].set_xlabel("x (arcsec)", fontsize=15)
    axes[0].set_ylabel("y (arcsec)", fontsize=15)

    plt.subplots_adjust(
        wspace=0.0, hspace=0.0, left=0.05, right=0.95, bottom=0.15, top=0.9
    )

    if return_axes:
        return axes
