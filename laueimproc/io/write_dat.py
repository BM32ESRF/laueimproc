#!/usr/bin/env python3

"""Write a dat file."""

import datetime
import pathlib
import typing

import torch

from laueimproc.classes.diagram import Diagram


def write_dat(filename: typing.Union[str, bytes, pathlib.Path], diagram: Diagram):
    """Write the dat file associate to the provided diagram.

    The file contains the following columns:

    * peak_X : float
        The x position of the center of the peak in convention j+1/2 in this module.
        It is like cv2 convention but start by (1, 1), not (0, 0).
        The position is the position estimated after a gaussian fit with mse criteria.
    * peak_Y : float
        The y position of the center of the peak in convention i+1/2 in this module.
        Same formalism as peak_X.
    * peak_Itot : float
        peak_Isub + the mean of the background estimation in the roi.
    * peak_Isub : float
        Gaussian amplitude, fitted on the roi with no background. Intensity without background.
    * peak_fwaxmaj : float
        Main full with. It is 2*std of the gaussian in the main axis of the PCA.
    * peak_fwaxmin : float
        Same as peak_fwaxmaj along the secondary axis.
    * peak_inclination : float
        The angle between the horizontal axis (i) and the main pca axis of the spot.
        The angle is defined in clockwise (-trigo_angle(i, j)) in order to give a positive angle
        when we plot the image and when we have a vien in the physician base (-j, i).
        The angle is defined between -90 and 90 degrees.
    * Xdev : float
        The difference between the position of the gaussian and the baricenter
        along the x axis in cv2 convention (axis j in this module).
    * Ydev : float
        Same as Xdev for the y axis.
    * peak_bkg : float
        The mean of the estimated background in the roi.
    * Ipixmax : int
        The max intensity of the all the pixel in the roi with no background.

    The peaks are written in the native order of the gaussian.

    Parameters
    ----------
    filename : pathlike
        The path to the file, relative or absolute.
        The suffix ".dat" is automaticaly append if it is not already provided.
        If a file is already existing, it is overwriten.
    diagram : laueimproc.classes.diagram.Diagram
        The reference to the diagram.

    Examples
    --------
    >>> import pathlib
    >>> import tempfile
    >>> from laueimproc.classes.diagram import Diagram
    >>> from laueimproc.io.write_dat import write_dat
    >>> import laueimproc.io
    >>> diagram = Diagram(pathlib.Path(laueimproc.io.__file__).parent / "ge_blanc.jp2")
    >>> diagram.find_spots()
    >>> file = pathlib.Path(tempfile.gettempdir()) / "ge_blanc.dat"
    >>> write_dat(file, diagram)
    >>> with open(file, "r", encoding="utf-8") as raw:
    ...     print(raw.read())
    ...
    >>>
    """
    # verification
    assert isinstance(filename, (str, bytes, pathlib.Path)), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve().with_suffix(".dat")
    assert isinstance(diagram, Diagram), diagram.__class__.__name__

    positions, _, magnitudes, infodict = diagram.fit_gaussian(eigtheta=True)
    positions = positions.squeeze(2)
    two_stds = 2 * torch.sqrt(infodict["eigtheta"][:, :2])
    thetas = torch.rad2deg(infodict["eigtheta"][:, 2])
    rawrois = diagram.rawrois
    barycenters = diagram.compute_barycenters()
    backgrounds = torch.sum(rawrois - diagram.rois, axis=(1, 2))
    backgrounds /= (diagram.bboxes[:, 2]*diagram.bboxes[:, 3]).to(backgrounds.dtype)
    pixmaxs = torch.amax(rawrois, axis=(1, 2))

    file_content = (
        "peak_X peak_Y "
        "peak_Itot peak_Isub "
        "peak_fwaxmaj peak_fwaxmin peak_inclination "
        "Xdev Ydev "
        "peak_bkg Ipixmax\n"
    )
    for (pos_i, pos_j), mag, bkg, (two_std1, two_std2), theta, (bar_i, bar_j), pixmax in zip(
        positions.tolist(),
        (65535*magnitudes).tolist(), (65535*backgrounds).tolist(),
        two_stds.tolist(), thetas.tolist(),
        barycenters.tolist(),
        (65535*pixmaxs).tolist(),
    ):
        # print(pos_i, pos_j, mag, bkg, two_std1, two_std2, theta, bar_i, bar_j, pixmax)
        file_content += (
            f"{pos_i+0.5:.3f} {pos_j+0.5:.3f} "
            f"{mag+bkg:.1f} {mag:.1f} "
            f"{two_std1:.3f} {two_std2:.3f} {theta:.1f} "
            f"{pos_i-bar_i:.3f} {pos_j-bar_j} "
            f"{bkg:.1f} {pixmax:.1f}\n"
        )
    file_content += (
        f"# file created by laueimproc from {diagram.file.name} "
        f"at {datetime.datetime.today().isoformat()}\n"
        f"# from the parent directory: {str(diagram.file.parent)}"
    )

    with open(filename, "w", encoding="utf-8") as file:
        file.write(file_content)
