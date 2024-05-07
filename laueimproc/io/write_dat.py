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
    >>> diagram = Diagram(pathlib.Path(laueimproc.io.__file__).parent / "ge.jp2")
    >>> diagram.find_spots()
    >>> file = pathlib.Path(tempfile.gettempdir()) / "ge_blanc.dat"
    >>> write_dat(file, diagram)
    >>>
    >>> with open(file, "r", encoding="utf-8") as raw:  # doctest: +ELLIPSIS
    ...     print(raw.read())
    ...
       peak_X    peak_Y   peak_Itot   peak_Isub peak_fwaxmaj ...ation   Xdev   Ydev peak_bkg Ipixmax
        5.549     4.764    164733.5    163837.5        7.054 ...-50.9  0.000  0.000    896.0  1121.0
       26.591     1.228     33794.9     32767.5        5.721 ...-87.9  0.000  0.000   1027.4  1129.0
    ...
     1974.962  1938.139 126975112.9 126974064.0        4.097 ... 28.1  0.000  0.000   1048.9  4999.0
     1179.912  1976.142 129465501.1 129464392.0        5.356 ...  5.7  0.000  0.000   1109.1 20258.0
    # file created by laueimproc ...
    # Diagram from ge.jp2:
    #     History:
    #         1. 240 spots from self.find_spots()
    #     No Properties
    #     Current state:
    #         * id, state: ...
    #         * nbr spots: 240
    #         * total mem: 16.4MB
    >>>
    """
    # verification
    assert isinstance(filename, (str, pathlib.Path)), filename.__class__.__name__
    filename = pathlib.Path(filename).expanduser().resolve().with_suffix(".dat")
    assert isinstance(diagram, Diagram), diagram.__class__.__name__

    # positions, _, magnitudes, infodict = diagram.fit_gaussian(eigtheta=True)
    # positions = positions.squeeze(2)
    positions, _, infodict = diagram.fit_gaussian_em(eigtheta=True)
    magnitudes = diagram.compute_rois_max()[:, 0]

    backgrounds = torch.sum(diagram.rawrois - diagram.rois, dim=(1, 2))
    backgrounds /= (diagram.bboxes[:, 2]*diagram.bboxes[:, 3]).to(backgrounds.dtype)

    # header
    file_content = (
        "   peak_X    peak_Y "
        "  peak_Itot   peak_Isub "
        "peak_fwaxmaj peak_fwaxmin peak_inclination "
        "  Xdev   Ydev "
        "peak_bkg Ipixmax\n"
    )

    # content
    for pos, mag, bkg, two_std, theta, bary, pixmax in zip(
        positions.tolist(),
        (65535*magnitudes).tolist(), (65535*backgrounds).tolist(),
        (2*torch.sqrt(infodict["eigtheta"][:, :2])).tolist(),
        torch.rad2deg(infodict["eigtheta"][:, 2]).tolist(),
        diagram.compute_rois_centroid().tolist(),
        (65535*torch.amax(diagram.rawrois, dim=(1, 2))).tolist(),
    ):
        file_content += (
            f"{pos[1]+0.5:9.3f} {pos[0]+0.5:9.3f} "  # ij to xy
            f"{mag+bkg:11.1f} {mag:11.1f} "
            f"{two_std[0]:12.3f} {two_std[1]:12.3f} {theta:16.1f} "
            f"{pos[1]-bary[1]:6.3f} {pos[0]-bary[0]:6.3f} "  # ij to xy
            f"{bkg:8.1f} {pixmax:7.1f}\n"
        )

    # comments
    file_content += (
        f"# file created by laueimproc {__version__} "
        f"at {datetime.datetime.today().isoformat()}\n"
    )
    file_content += "# " + str(diagram).replace("\n", "\n# ")

    with open(filename, "w", encoding="utf-8") as file:
        file.write(file_content)
