"""
Overview: Galaxies
------------------

Nearly a century ago, Edwin Hubble famously classified galaxies into three distinct groups: ellipticals, spirals and
irregulars. Today, by analysing millions of galaxies with advanced image processing techniques Astronomers have
expanded on this picture and revealed the rich diversity of galaxy morphology both in the nearby and distant
Universe.

`PyAutoGalaxy` is an open-source Python package for the multi-wavelength analysis of the morphologies of large
galaxy samples.

To begin, lets import autogalaxy and the plot module.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from astropy import cosmology as cosmo
import autogalaxy as ag
import autogalaxy.plot as aplt

import os

workspace_path = os.getcwd()

"""
__Grids__

To describe the luminous emission of galaxies, **PyAutoGalaxy** uses `Grid2D` data structures, which are 
two-dimensional Cartesian grids of (y,x) coordinates. 

Below, we make and plot a uniform Cartesian grid:
"""
grid_2d = ag.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

grid_2d_plotter = aplt.Grid2DPlotter(
    grid=grid_2d,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="grid_2d", format="png")
    ),
)
grid_2d_plotter.figure_2d()

"""
__Light Profiles__

We will use this `Grid2D`'s coordinates to evaluate the galaxy's morphology. We therefore need analytic 
functions representing a galaxy's light distribution(s). 

For this,  **PyAutoGalaxy** uses `LightProfile` objects, for example the `Sersic` `LightProfile` object which
represents a light distribution:
"""
sersic_light_profile = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.2, 0.1),
    intensity=0.005,
    effective_radius=2.0,
    sersic_index=4.0,
)

"""
By passing this profile a `Grid2D`, we evaluate the light at every (y,x) coordinate on the `Grid2D` and create an 
image of the `LightProfile`.
"""
image_2d = sersic_light_profile.image_2d_from(grid=grid_2d)

"""
The PyAutoGalaxy plot module provides methods for plotting objects and their properties, like 
the `LightProfile`'s image.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile,
    grid=grid_2d,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="sersic_light_profile", format="png"
        )
    ),
)
light_profile_plotter.figures_2d(image=True)

"""
__Galaxies__

A `Galaxy` object is a collection of `LightProfile` objects at a given redshift. 

The code below creates a galaxy which is made of two components, a bulge and disk.
"""
bulge = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

disk = ag.lp.Exponential(
    centre=(0.0, 0.0),
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    intensity=0.5,
    effective_radius=1.6,
)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge, disk=disk)

"""
We can create an image the galaxy by passing it the 2D grid above.
"""
image_2d = galaxy.image_2d_from(grid=grid_2d)

"""
The **PyAutoGalaxy** plot module provides methods for plotting galaxies.

Below, we plot its image, which is the sum of the bulge and disk components.
"""
galaxy_plotter = aplt.GalaxyPlotter(
    galaxy=galaxy,
    grid=grid_2d,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="galaxy", format="png")
    ),
)
galaxy_plotter.figures_2d(image=True)

"""
__Galaxies__

If our observation contains multiple galaxies, we create a `Plane` object to represent all galaxies.

By passing `Galaxy` objects to a `Plane`, **PyAutoGalaxy** groups them to indicate they are at the same redshift.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, -1.0),
        ell_comps=(0.25, 0.1),
        intensity=0.1,
        effective_radius=0.8,
        sersic_index=2.5,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 1.0),
        ell_comps=(0.0, 0.1),
        intensity=0.1,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

"""
The image of the plane consists of all galaxies. 

**PyAutoGalaxy** plot tools allow us to plot this image or a subplot containing images of each individual galaxy.
"""
image_2d = plane.image_2d_from(grid=grid_2d)

plane_plotter = aplt.GalaxiesPlotter(
    plane=plane,
    grid=grid_2d,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="plane", format="png")
    ),
)
plane_plotter.figures_2d(image=True)
plane_plotter = aplt.GalaxiesPlotter(
    plane=plane,
    grid=grid_2d,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="subplot_galaxies", format="png"
        )
    ),
)
plane_plotter.subplot_galaxy_images()

"""
__Extending Objects__

The PyAutoGalaxy API isn designed such that all of the objects introduced above are extensible. `Galaxy` objects 
can take many `LightProfile`'s and `Plane`'s many `Galaxy`'s. 

To finish, lets create a `Plane` with 2 merging galaxies, where the second galaxy has multiple star forming clumps.
"""
galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lmp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05),
        intensity=0.3,
        effective_radius=0.3,
        sersic_index=3.5,
        mass_to_light_ratio=0.6,
    ),
    disk=ag.lmp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.1),
        intensity=0.5,
        effective_radius=2.0,
        mass_to_light_ratio=0.2,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=1.0,
    bulge=ag.lp.Exponential(
        centre=(2.0, 2.0),
        ell_comps=(0.05, 0.05),
        intensity=1.2,
        effective_radius=0.8,
    ),
    clump_0=ag.lp.Sersic(centre=(3.0, 3.0), intensity=0.5, effective_radius=0.2),
    clump_1=ag.lp.Sersic(centre=(2.5, 2.8), intensity=0.5, effective_radius=0.2),
    clump_2=ag.lp.Sersic(centre=(1.7, 1.3), intensity=0.5, effective_radius=0.2),
)

plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

"""
This is what the merging galaxies look like:
"""
plane_plotter = aplt.GalaxiesPlotter(
    plane=plane,
    grid=grid_2d,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="merging_galaxies", format="png"
        )
    ),
)
plane_plotter.figures_2d(image=True)

"""
__Wrap Up__

A more detailed description of these **PyAutoGalaxy** objects is given in chapter 1 of the **HowToGalaxy** 
tutorials, which I strongly advise new users check out!
"""
