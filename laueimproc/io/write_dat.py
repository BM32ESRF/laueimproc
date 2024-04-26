#!/usr/bin/env python3

"""Write a dat file."""

import datetime
import pathlib
import typing

import torch

from laueimproc.classes.diagram import Diagram
from laueimproc import __version__


def write_dat(filename: typing.Union[str, pathlib.Path], diagram: Diagram):
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
    >>> with open(file, "r", encoding="utf-8") as raw:  # doctest: +ELLIPSIS
    ...     print(raw.read())
    ...
    peak_X peak_Y peak_Itot peak_Isub ... peak_inclination Xdev Ydev peak_bkg Ipixmax
    1987.658 896.926 57.0 27.0 2.682 2.561 62.9 0.000 0.0 30.0 57.0
    1945.967 914.329 5263.6 5223.0 4.403 4.025 -20.4 0.000 0.0 40.6 5264.0
    ...
    19.785 1209.065 48.7 30.0 2.781 1.585 -80.7 0.000 0.0 18.7 55.0
    9.779 904.985 50.8 29.0 2.682 1.720 82.0 0.000 0.0 21.8 58.0
    # file created by laueimproc ...
    # Diagram from ge_blanc.jp2:
    #     History:
    #         1. 781 spots from self.find_spots()
    #     No Properties
    #     Current state:
    #         * id, state: ...
    #         * nbr spots: 781
    #         * total mem: 16.9MB
    >>>
    """
    # verification
    assert isinstance(filename, (str, pathlib.Path)), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve().with_suffix(".dat")
    assert isinstance(diagram, Diagram), diagram.__class__.__name__

    # positions, _, magnitudes, infodict = diagram.fit_gaussian(eigtheta=True)
    # positions = positions.squeeze(2)
    positions, _, infodict = diagram.fit_gaussian_em(eigtheta=True)
    magnitudes = torch.amax(diagram.rois, dim=(1, 2))

    backgrounds = torch.sum(diagram.rawrois - diagram.rois, dim=(1, 2))
    backgrounds /= (diagram.bboxes[:, 2]*diagram.bboxes[:, 3]).to(backgrounds.dtype)

    # header
    file_content = (
        "peak_X peak_Y "
        "peak_Itot peak_Isub "
        "peak_fwaxmaj peak_fwaxmin peak_inclination "
        "Xdev Ydev "
        "peak_bkg Ipixmax\n"
    )

    # content
    for pos, mag, bkg, two_std, theta, bary, pixmax in zip(
        positions.tolist(),
        (65535*magnitudes).tolist(), (65535*backgrounds).tolist(),
        (2*torch.sqrt(infodict["eigtheta"][:, :2])).tolist(),
        torch.rad2deg(infodict["eigtheta"][:, 2]).tolist(),
        diagram.compute_barycenters().tolist(),
        (65535*torch.amax(diagram.rawrois, dim=(1, 2))).tolist(),
    ):
        file_content += (
            f"{pos[0]+0.5:.3f} {pos[1]+0.5:.3f} "
            f"{mag+bkg:.1f} {mag:.1f} "
            f"{two_std[0]:.3f} {two_std[1]:.3f} {theta:.1f} "
            f"{pos[0]-bary[0]:.3f} {pos[1]-bary[1]} "
            f"{bkg:.1f} {pixmax:.1f}\n"
        )

    # comments
    file_content += (
        f"# file created by laueimproc {__version__} "
        f"at {datetime.datetime.today().isoformat()}\n"
    )
    file_content += "# " + str(diagram).replace("\n", "\n# ")

    with open(filename, "w", encoding="utf-8") as file:
        file.write(file_content)
