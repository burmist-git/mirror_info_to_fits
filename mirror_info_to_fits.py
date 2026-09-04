#!/usr/bin/python3
"""
Read a mirror list file and write mirror ID, position, diameter, normal, and shape to a
FITS table.

Reuses simtools.model.mirrors.Mirrors for reading the mirror list file (sim_telarray
or ecsv format) and simtools.visualization.plot_mirrors.get_mirror_positions_and_ids
for extracting the mirror positions and IDs, the same way the mirror layout plot does.
The Z position and the mirror normal vector (nx, ny, nz) are computed from the
telescope type and the focal length read from the parameters file (see get_z_pos
and get_nx_ny_nz). Also plots the mirror profile (z vs. radius) to a PDF file
next to the FITS output, using
simtools.visualization.plot_mirrors.plot_mirror_profile.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.io import ascii_handler, table_handler
from simtools.model.mirrors import Mirrors
from simtools.visualization import visualize
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_mirrors import (
    get_mirror_positions_and_ids,
    get_ring_segment_positions_and_ids,
    get_shape_segment_positions_and_ids,
    plot_mirror_layout,
    plot_mirror_profile,
    plot_mirror_ring_segmentation,
    plot_mirror_shape_segmentation,
)

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mirror_list_file", type=Path, help="Path to the mirror list file.")
    parser.add_argument("output_file", type=Path, help="Path to the output FITS file.")
    parser.add_argument(
        "parameters_file",
        type=Path,
        help=(
            "JSON file with primary mirror parameters (e.g., mirror_panel_diameter, "
            "mirror_focal_length, mirror_panel_shape)"
        ),
    )
    parser.add_argument(
        "tel_type",
        type=str,
        choices=["LST", "MST", "SST"],
        help="Telescope type used to compute the mirror z position (LST, MST, or SST).",
    )
    parser.add_argument(
        "--ring_segmentation",
        action="store_true",
        help=(
            "Treat mirror_list_file as a ring segmentation file (lines of the form "
            "'ring <nseg> <rmin> <rmax> <dphi> <phi0>') and use "
            "get_ring_segment_positions_and_ids instead of reading a mirror list file."
        ),
    )
    parser.add_argument(
        "--shape_segmentation",
        action="store_true",
        help=(
            "Treat mirror_list_file as a shape segmentation file (lines of the form "
            "'Shape <ID> <Xc> <Yc> <Diam> <Rot>'"
            "get_shape_segment_positions_and_ids instead of reading a mirror list file."
        ),
    )
    return parser.parse_args()


def read_parameters_file(parameters_file):
    """
    Read a JSON file with model parameters.

    Parameters
    ----------
    parameters_file : str or Path
        Path to the JSON parameters file.

    Returns
    -------
    dict
        Parameters dictionary read from the JSON file.
    """
    return ascii_handler.collect_data_from_file(parameters_file)


def rotate_xy(x_pos, y_pos, angle):
    """
    Rotate X/Y positions counter-clockwise by an arbitrary angle.

    Parameters
    ----------
    x_pos : numpy.ndarray
        X positions.
    y_pos : numpy.ndarray
        Y positions.
    angle : float
        Rotation angle in radians (counter-clockwise).

    Returns
    -------
    tuple
        (x_rot, y_rot): rotated X and Y positions (numpy arrays).
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = x_pos * cos_a - y_pos * sin_a
    y_rot = x_pos * sin_a + y_pos * cos_a
    return x_rot, y_rot


def get_z_pos(tel_type, parameters, x_pos, y_pos):
    """
    Compute the mirror panel z position from the telescope type and focal length.

    Parameters
    ----------
    tel_type : str
        Telescope type, one of "LST", "MST", "SST".
    parameters : dict
        Model parameters dictionary, as read by read_parameters_file().
    x_pos : numpy.ndarray
        Mirror panel X positions.
    y_pos : numpy.ndarray
        Mirror panel Y positions.

    Returns
    -------
    numpy.ndarray or float
        Z position(s) of the mirror panels.
    """
    r_axis = np.sqrt(x_pos**2 + y_pos**2)

    if tel_type == "LST":
        # Paraboloid sag: z(r) = r**2 / (4 * focal_length), zero at the dish vertex.
        focal_length = parameters["focal_length"]
        z_pos = r_axis**2 / (4 * focal_length)

    if tel_type == "MST":
        fshape = parameters["value"]
        z_pos = fshape - np.sqrt(fshape * fshape - r_axis * r_axis)

    if tel_type == "SST":
        # Dish shape polynomial: z(r) = sum_i value[i] * r**(2 * i).
        coefficients = parameters["value"]
        z_pos = np.zeros_like(r_axis, dtype=float)
        for i, coefficient in enumerate(coefficients):
            z_pos += coefficient * r_axis ** (2 * i)

    return z_pos


def get_nx_ny_nz(tel_type, parameters, x_pos, y_pos, z_pos):
    """
    Compute the mirror panel normal vector components.

    Parameters
    ----------
    tel_type : str
        Telescope type, one of "LST", "MST", "SST".
    parameters : dict
        Model parameters dictionary, as read by read_parameters_file().
    x_pos : numpy.ndarray
        Mirror panel X positions.
    y_pos : numpy.ndarray
        Mirror panel Y positions.
    z_pos : numpy.ndarray
        Mirror panel Z positions.

    Returns
    -------
    tuple
        (nx, ny, nz): normal vector components (numpy arrays), one per mirror panel.
    """
    nx = np.zeros_like(x_pos, dtype=float)
    ny = np.zeros_like(x_pos, dtype=float)
    nz = np.zeros_like(x_pos, dtype=float)

    r_axis = np.sqrt(x_pos**2 + y_pos**2)
    

    if tel_type == "LST":
        # Surface normal of the paraboloid z(r) = r**2 / (4 * focal_length):
        # N = (-dz/dx, -dz/dy, 1), normalized, with dz/dr = r / (2 * focal_length).
        focal_length = parameters["focal_length"]
        dz_dr = r_axis / (2 * focal_length)


    if tel_type == "MST":
        # Surface normal of the paraboloid z(r) = fshape - sqt(fshape**2 - r**2):
        # N = (-dz/dx, -dz/dy, 1), normalized, with dz/dr = r / sqt(fshape**2 - r**2).
        fshape = parameters["value"]
        dz_dr = r_axis / np.sqrt(fshape * fshape - r_axis * r_axis)

        
    if tel_type == "SST":
        # Surface normal of the dish shape z(r) = sum_i value[i] * r**(2 * i):
        # N = (-dz/dx, -dz/dy, 1), normalized, with dz/dr = sum_i value[i] * 2*i * r**(2*i - 1).
        coefficients = parameters["value"]
        dz_dr = np.zeros_like(r_axis, dtype=float)
        for i, coefficient in enumerate(coefficients):
            if i > 0:
                dz_dr += coefficient * 2 * i * r_axis ** (2 * i - 1)


    cos_phi = np.divide(x_pos, r_axis, out=np.zeros_like(r_axis, dtype=float), where=r_axis > 0)
    sin_phi = np.divide(y_pos, r_axis, out=np.zeros_like(r_axis, dtype=float), where=r_axis > 0)
    norm = np.sqrt(dz_dr**2 + 1)

    nx = -dz_dr * cos_phi / norm
    ny = -dz_dr * sin_phi / norm
    nz = 1 / norm

        
    return nx, ny, nz


def get_normal_vec_with_axis_z(tel_type, parameters, x_pos, y_pos, z_pos, nz):
    """
    Compute the mirror normal's angle to the z axis in two independent ways.

    Method 1 (ray optics): the angle between the z axis and the ray from the mirror
    panel to the focus point, arctan(r_axis / (focal_length - z_pos)), halved per the
    law of reflection (the normal bisects the incoming ray, parallel to the z axis, and
    the ray reflected towards the focus). This is exact for a true paraboloid (LST);
    for MST, whose dish is modeled as spherical rather than parabolic, it is only an
    approximation. Not defined for SST, which has no single focal_length parameter
    (its dish shape is a general polynomial); alpha_focus is 0 in that case.

    Method 2 (differential geometry): arccos(nz), directly from the z component of the
    already-computed unit surface normal (see get_nx_ny_nz).

    For LST and MST these two methods should agree to numerical precision (LST exactly,
    MST approximately, since the reflection-to-focus construction is only exact for a
    true parabola); comparing them is a cross-check of the normal computation.

    Parameters
    ----------
    tel_type : str
        Telescope type, one of "LST", "MST", "SST".
    parameters : dict
        Model parameters dictionary, as read by read_parameters_file().
    x_pos : numpy.ndarray
        Mirror panel X positions.
    y_pos : numpy.ndarray
        Mirror panel Y positions.
    z_pos : numpy.ndarray
        Mirror panel Z positions.
    nz : numpy.ndarray
        Z component of the unit surface normal, as returned by get_nx_ny_nz.

    Returns
    -------
    tuple
        (alpha_focus, alpha_normal): normal-to-z-axis angle in radians, computed via
        the ray-optics focus construction and via arccos(nz), respectively.
    """
    r_axis = np.sqrt(x_pos**2 + y_pos**2)

    alpha_focus = np.zeros_like(r_axis, dtype=float)

    if tel_type == "LST":
        focal_length = parameters["focal_length"]
        with np.errstate(divide="ignore"):
            alpha_focus = np.pi / 2.0 - np.arctan((focal_length - z_pos) / r_axis)
        alpha_focus /= 2

    if tel_type == "MST":
        fshape = parameters["value"]
        focal_length = fshape / 1.2
        with np.errstate(divide="ignore"):
            alpha_focus = np.pi / 2.0 - np.arctan((focal_length - z_pos) / r_axis)
        alpha_focus /= 2

    if tel_type == "SST":
        focal_length = 415
        with np.errstate(divide="ignore"):
            alpha_focus = np.pi / 2.0 - np.arctan((focal_length - z_pos) / r_axis)
        alpha_focus /= 2

    return alpha_focus, np.arccos(nz)


def get_hexagon_surface(diameter):
    """
    Compute the surface area of a regular hexagonal mirror panel.

    The panel diameter is the flat-to-flat width, matching the convention used
    for the hexagon patches in plot_mirrors._create_single_mirror_patch (whose
    circumradius is diameter / sqrt(3)); for a regular hexagon the side length
    equals the circumradius, giving area = sqrt(3) / 2 * diameter**2.

    Parameters
    ----------
    diameter : numpy.ndarray or float
        Mirror panel diameter (flat-to-flat width).

    Returns
    -------
    numpy.ndarray or float
        Hexagon surface area, in the same length unit squared as diameter.
    """
    return np.sqrt(3) / 2 * diameter**2


def plot_mirror_positions(x_pos, y_pos, telescope_model_name=None):
    """
    Plot the mirror panel X-Y positions as a plain scatter (no panel shapes).

    Parameters
    ----------
    x_pos : numpy.ndarray
        X position of each mirror panel.
    y_pos : numpy.ndarray
        Y position of each mirror panel.
    telescope_model_name : str, optional
        Name of the telescope model, used in the plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    logger.info(f"Plotting mirror x-y positions for {telescope_model_name}")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x_pos, y_pos, s=10, alpha=0.8, edgecolor="black", facecolor="dodgerblue")
    ax.set_xlabel("X position [cm]", fontsize=12)
    ax.set_ylabel("Y position [cm]", fontsize=12)
    title = "Mirror positions"
    if telescope_model_name:
        title += f" - {telescope_model_name}"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    return fig


def plot_mirror_facets(
    mirror_list_file,
    tel_type,
    mirrors=None,
    ring_segmentation=False,
    shape_segmentation=False,
):
    """
    Plot the mirror facet layout, reusing the plot_mirrors layout plotting functions.

    Parameters
    ----------
    mirror_list_file : str or Path
        Path to the mirror list file, or, if ring_segmentation or shape_segmentation is
        True, path to the corresponding segmentation file.
    tel_type : str
        Telescope type, used as the plot title.
    mirrors : Mirrors, optional
        Mirrors object, required unless ring_segmentation or shape_segmentation is True.
    ring_segmentation : bool, optional
        If True, plot mirror_list_file as a ring segmentation file.
    shape_segmentation : bool, optional
        If True, plot mirror_list_file as a shape segmentation file.

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure object, or None if no facet data could be plotted
        (e.g. no ring data found in a ring segmentation file).
    """
    if ring_segmentation:
        return plot_mirror_ring_segmentation(
            data_file_path=mirror_list_file,
            telescope_model_name=tel_type,
            parameter_type="primary_mirror_segmentation",
        )

    if shape_segmentation:
        return plot_mirror_shape_segmentation(
            data_file_path=mirror_list_file,
            telescope_model_name=tel_type,
            parameter_type="primary_mirror_segmentation",
        )

    return plot_mirror_layout(
        mirrors=mirrors,
        mirror_file_path=mirror_list_file,
        telescope_model_name=tel_type,
    )


def mirror_table_to_fits(
    mirror_list_file,
    output_file,
    parameters_file,
    tel_type,
    ring_segmentation=False,
    shape_segmentation=False,
):
    """
    Read a mirror list file and write mirror ID, position, diameter, and normal to a FITS file.

    Parameters
    ----------
    mirror_list_file : str or Path
        Path to the mirror list file (sim_telarray or ecsv format), or, if
        ring_segmentation is True, path to a ring segmentation file.
    output_file : str or Path
        Path to the output FITS file.
    parameters_file : str or Path, optional
        Path to a JSON file with model parameters, used to compute the mirror
        panel Z position (see get_z_pos).
    tel_type : str
        Telescope type ("LST", "MST", or "SST"), used to compute the mirror
        panel Z position (see get_z_pos).
    ring_segmentation : bool, optional
        If True, read mirror_list_file as a ring segmentation file and use
        get_ring_segment_positions_and_ids instead of reading a mirror list file.
    shape_segmentation : bool, optional
        If True, read mirror_list_file as a shape segmentation file and use
        get_shape_segment_positions_and_ids instead of reading a mirror list file.
    """
    parameters = read_parameters_file(parameters_file)
    logger.info(f"Read parameters from {parameters_file}: {parameters}")

    if ring_segmentation:
        x_pos, y_pos, mirror_ids = get_ring_segment_positions_and_ids(mirror_list_file)
        # Ring segmentation files define segments by (rmin, rmax), not a panel diameter.
        diameter = np.full(len(mirror_ids), np.nan)
        position_unit = None
    elif shape_segmentation:
        x_pos, y_pos, diameter, mirror_ids = get_shape_segment_positions_and_ids(mirror_list_file)
        position_unit = None
    else:
        mirrors = Mirrors(mirror_list_file=mirror_list_file)
        x_pos, y_pos, diameter, mirror_ids = get_mirror_positions_and_ids(mirrors)
        print("tel_type: = ", tel_type)
        if tel_type == "LST":
            x_pos = x_pos - diameter / 2.0 / np.sqrt(3)
        position_unit = "cm"


    x_pos, y_pos = rotate_xy(x_pos, y_pos, np.pi / 2)
        
        
    area_unit = "cm2" if position_unit == "cm" else None

    z_pos = get_z_pos(tel_type, parameters, x_pos, y_pos)
    nx, ny, nz = get_nx_ny_nz(tel_type, parameters, x_pos, y_pos, z_pos)
    surface = get_hexagon_surface(diameter)
    alpha_focus, alpha_normal = get_normal_vec_with_axis_z(
        tel_type, parameters, x_pos, y_pos, z_pos, nz
    )

    # "diameter": diameter,
    # position_unit,  # diameter
    # "angle_z_focus": alpha_focus,
    # "angle_z_normal": alpha_normal,
    # "rad",  # angle_z_focus
    # "rad",  # angle_z_normal
    
    table = Table(
        {
            "mirror_id": list(mirror_ids),
            "x": x_pos,
            "y": y_pos,
            "z": z_pos,
            "surface": surface,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "shape": ["HEXAGON"] * len(mirror_ids),
        },
        units=[
            None,  # mirror_id
            position_unit,  # x
            position_unit,  # y
            position_unit,  # z
            area_unit,  # surface
            None,  # nx
            None,  # ny
            None,  # nz
            None,  # shape
        ],
    )
    table.meta["EXTNAME"] = "MIRRORS"

    output_file = Path(output_file)
    table_handler.write_tables([table], output_file, overwrite_existing=True, file_type="FITS")
    logger.info(f"Wrote {len(table)} mirror positions from {mirror_list_file} to {output_file}")

    r_pos = np.sqrt(x_pos**2 + y_pos**2)
    fig = plot_mirror_profile(r_pos, z_pos, telescope_model_name=tel_type)
    pdf_file = output_file.with_suffix(".pdf")
    visualize.save_figure(
        fig, pdf_file, figure_format=["pdf"], log_title="mirror profile", close=True
    )

    positions_fig = plot_mirror_positions(x_pos, y_pos, telescope_model_name=tel_type)
    positions_pdf_file = output_file.with_name(f"{output_file.stem}_positions.pdf")
    visualize.save_figure(
        positions_fig,
        positions_pdf_file,
        figure_format=["pdf"],
        log_title="mirror x-y positions",
        close=True,
    )

    facets_fig = plot_mirror_facets(
        mirror_list_file,
        tel_type,
        mirrors=mirrors if not (ring_segmentation or shape_segmentation) else None,
        ring_segmentation=ring_segmentation,
        shape_segmentation=shape_segmentation,
    )
    if facets_fig is not None:
        facets_pdf_file = output_file.with_name(f"{output_file.stem}_facets.pdf")
        visualize.save_figure(
            facets_fig,
            facets_pdf_file,
            figure_format=["pdf"],
            log_title="mirror facets",
            close=True,
        )


def main():
    """Read a mirror list file and write mirror positions to a FITS file."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    mirror_table_to_fits(
        args.mirror_list_file,
        args.output_file,
        args.parameters_file,
        args.tel_type,
        args.ring_segmentation,
        args.shape_segmentation,
    )


if __name__ == "__main__":
    main()
