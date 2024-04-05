#!/usr/bin/env python3

"""Write a dat file."""

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
        The difference between the argmax of the gaussian and the baricenter
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
    filename = pathlib.Path(filename).expanduser().resolve()
    assert isinstance(diagram, Diagram), diagram.__class__.__name__

    filename = filename.with_suffix(".dat")
    position, _, infodict = diagram.fit_gaussian_em(eigtheta=True)
    mean_mass = diagram.compute_pxl_intensities()
    mean_mass /= (diagram.bboxes[:, 2]*diagram.bboxes[:, 3]).to(mean_mass.dtype) * 65535
    barycenters = diagram.compute_barycenters()

    with open(filename, "w", encoding="utf-8") as file:
        file.write(
            "peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin "
            "peak_inclination Xdev Ydev peak_bkg Ipixmax\n"
        )
        for i, spot in enumerate(diagram.spots):
            roi = spot.roi * 65535
            bkg = spot.rawroi*65535 - roi
            # peak_X peak_Y
            file.write(f"{float(position[i][0]+0.5):.2f} {float(position[i][1]+0.5)} ")
            # peak_Itot peak_Isub
            file.write(f"{float(mean_mass[i]+bkg.mean()):.4f} {float(mean_mass[i]):.4f} ")
            # peak_fwaxmaj peak_fwaxmin
            file.write(
                f"{float(2*torch.sqrt(infodict['eigtheta'][i, 0])):.3f} "
                f"{float(2*torch.sqrt(infodict['eigtheta'][i, 1])):.3f} "
            )
            # peak_inclination
            file.write(f"{float(-torch.rad2deg(infodict['eigtheta'][i, 2])):.1f} ")
            # Xdev Ydev
            file.write(
                f"{float(position[i][1]-barycenters[i][1]):.2f} "
                f"{float(position[i][0]-barycenters[i][0]):.2f} "
            )
            # peak_bkg Ipixmax
            file.write(f"{float(bkg.mean()):.4f} {float(roi.max()):.4f}\n")
