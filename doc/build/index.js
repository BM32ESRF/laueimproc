URLS=[
"laueimproc/index.html",
"laueimproc/classes/base_diagram.html",
"laueimproc/classes/base_dataset.html",
"laueimproc/io/index.html",
"laueimproc/io/write_dat.html",
"laueimproc/io/write.html",
"laueimproc/io/save_dataset.html",
"laueimproc/io/read.html",
"laueimproc/io/download.html",
"laueimproc/io/convert.html",
"laueimproc/io/comp_lm.html",
"laueimproc/improc/index.html",
"laueimproc/improc/spot/index.html",
"laueimproc/improc/spot/basic.html",
"laueimproc/improc/spot/pca.html",
"laueimproc/improc/spot/fit.html",
"laueimproc/improc/spot/extrema.html",
"laueimproc/improc/morpho.html",
"laueimproc/improc/find_bboxes.html",
"laueimproc/improc/peaks_search.html",
"laueimproc/classes/index.html",
"laueimproc/classes/diagram.html",
"laueimproc/classes/dataset.html",
"laueimproc/opti/index.html",
"laueimproc/opti/singleton.html",
"laueimproc/opti/memory.html",
"laueimproc/opti/rois.html",
"laueimproc/opti/gpu.html",
"laueimproc/opti/cache.html",
"laueimproc/gmm/index.html",
"laueimproc/gmm/check.html",
"laueimproc/gmm/gauss.html",
"laueimproc/gmm/fit.html",
"laueimproc/gmm/metric.html",
"laueimproc/gmm/linalg.html",
"laueimproc/gmm/gmm.html",
"laueimproc/testing/index.html",
"laueimproc/testing/install.html",
"laueimproc/testing/tests/index.html",
"laueimproc/testing/tests/inter_batch.html",
"laueimproc/testing/tests/set_spots.html",
"laueimproc/testing/tests/bragg.html",
"laueimproc/testing/tests/lattice.html",
"laueimproc/testing/tests/metric.html",
"laueimproc/testing/tests/projection.html",
"laueimproc/testing/tests/reciprocal.html",
"laueimproc/testing/tests/rotation.html",
"laueimproc/testing/run.html",
"laueimproc/testing/coding_style.html",
"laueimproc/nn/index.html",
"laueimproc/nn/dataaug/index.html",
"laueimproc/nn/dataaug/scale.html",
"laueimproc/nn/dataaug/patch.html",
"laueimproc/nn/train.html",
"laueimproc/nn/loader.html",
"laueimproc/nn/vae_spot_classifier.html",
"laueimproc/ml/index.html",
"laueimproc/ml/spot_dist.html",
"laueimproc/ml/dataset_dist.html",
"laueimproc/immix/index.html",
"laueimproc/immix/mean.html",
"laueimproc/immix/inter.html",
"laueimproc/common.html",
"laueimproc/geometry/index.html",
"laueimproc/geometry/metric.html",
"laueimproc/geometry/bragg.html",
"laueimproc/geometry/reciprocal.html",
"laueimproc/geometry/projection.html",
"laueimproc/geometry/lattice.html",
"laueimproc/geometry/rotation.html",
"laueimproc/geometry/thetachi.html",
"laueimproc/geometry/hkl.html",
"laueimproc/geometry/model.html",
"laueimproc/convention.html"
];
INDEX=[
{
"ref":"laueimproc",
"url":0,
"doc":"Group some image processing tools to annalyse Laue diagrams. [main documentation]( /html/index.html)"
},
{
"ref":"laueimproc.collect",
"url":0,
"doc":"Release all unreachable diagrams. Returns    - nbr : int The number of diagrams juste released.",
"func":1
},
{
"ref":"laueimproc.Diagram",
"url":0,
"doc":"A Laue diagram image. Create a new diagram with appropriated metadata. Parameters      data : pathlike or arraylike The filename or the array/tensor use as a diagram. For memory management, it is better to provide a pathlike rather than an array."
},
{
"ref":"laueimproc.Diagram.compute_rois_centroid",
"url":0,
"doc":"Compute the barycenter of each spots. Returns    - positions : torch.Tensor The 2 barycenter position for each roi. Each line corresponds to a spot and each column to an axis (shape (n, 2 . See  laueimproc.improc.spot.basic.compute_rois_centroid for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_centroid(  doctset: +ELLIPSIS tensor( 4.2637e+00, 5.0494e+00], [7.2805e-01, 2.6091e+01], [5.1465e-01, 1.9546e+03],  ., [1.9119e+03, 1.8759e+02], [1.9376e+03, 1.9745e+03], [1.9756e+03, 1.1794e+03 ) >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.compute_rois_inertia",
"url":0,
"doc":"Compute the inertia of the spot along the center of mass. Returns    - inertia : torch.Tensor The order 2 momentum centered along the barycenter (shape (n, . See  laueimproc.improc.spot.basic.compute_rois_inertia for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_inertia(  doctset: +ELLIPSIS tensor([4.0506e+00, 9.7974e-02, 1.6362e-01, 2.1423e-03, 4.8515e-02, 2.6657e-01, 9.9830e-02, 1.7221e-03, 2.8771e-02, 5.3754e-03, 2.4759e-02, 8.7665e-01, 5.1118e-02, 9.6223e-02, 3.0648e-01, 1.1362e-01, 2.6335e-01, 1.1388e-01,  ., 4.8586e+00, 4.4373e+00, 1.3409e-01, 3.1009e-01, 1.5538e-02, 8.4468e-03, 1.5235e+00, 1.0291e+01, 8.1125e+00, 5.4182e-01, 1.4235e+00, 4.4005e+00, 7.0188e+00, 2.5770e+00, 1.1837e+01, 7.0610e+00, 3.6725e+00, 2.2335e+01]) >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.compute_rois_max",
"url":0,
"doc":"Get the intensity and the position of the hottest pixel for each roi. Returns    - pos1_pos2_imax : torch.Tensor The concatenation of the colum vectors of the argmax and the intensity (shape (n, 3 . See  laueimproc.improc.spot.basic.compute_rois_max for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_max(  doctset: +ELLIPSIS tensor( 2.5000e+00, 5.5000e+00, 9.2165e-03], [5.0000e-01, 2.2500e+01, 1.3581e-03], [5.0000e-01, 1.9545e+03, 1.3123e-02],  ., [1.9125e+03, 1.8750e+02, 1.1250e-01], [1.9375e+03, 1.9745e+03, 6.0273e-02], [1.9755e+03, 1.1795e+03, 2.9212e-01 ) >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.compute_rois_nb_peaks",
"url":0,
"doc":"Find the number of extremums in each roi. Returns    - nbr_of_peaks : torch.Tensor The number of extremums (shape (n, . See  laueimproc.improc.spot.extrema.compute_rois_nb_peaks for more details. Notes   - No noise filtering. Doesn't detect shoulders. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_nb_peaks( tensor([23, 4, 3, 2, 6, 3, 1, 3, 1, 4, 1, 2, 3, 1, 1, 3, 3, 3, 2, 7, 4, 1, 1, 3, 3, 2, 4, 2, 1, 2, 1, 2, 2, 2, 2, 2, 6, 2, 1, 2, 4, 2, 3, 4, 1, 1, 4, 3, 1, 4, 4, 5, 3, 3, 1, 5, 3, 2, 6, 1, 2, 3, 1, 4, 6, 3, 6, 2, 2, 5, 7, 5, 3, 2, 3, 1, 2, 6, 8, 2, 3, 3, 3, 3, 2, 4, 2, 1, 1, 4, 4, 1, 4, 2, 5, 4, 3, 3, 1, 2, 1, 3, 4, 2, 5, 5, 7, 2, 6, 3, 5, 2, 2, 2, 7, 2, 3, 5, 3, 1, 4, 8, 3, 3, 4, 3, 4, 2, 5, 1, 7, 3, 5, 4, 10, 2, 3, 3, 4, 3, 5, 8, 2, 5, 8, 3, 2, 3, 5, 5, 5, 3, 4, 6, 2, 4, 1, 3, 4, 2, 3, 4, 5, 5, 2, 4, 6, 4, 4, 1, 2, 4, 4, 3, 13, 13, 3, 6, 3, 2, 1, 4, 6, 3, 2, 2, 16, 3, 1, 2, 2, 4, 1, 3, 7, 4, 1, 5, 4, 7, 5, 1, 6, 6, 2, 4, 3, 9, 3, 2, 3, 2, 8, 5, 3, 3, 3, 3, 10, 9, 2, 2, 5, 4, 3, 3, 2, 1, 3, 6, 11, 2, 4, 4, 6, 2, 7, 5, 2, 10], dtype=torch.int16) >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.compute_rois_pca",
"url":0,
"doc":"Compute the pca on each spot. Returns    - std1_std2_theta : torch.Tensor The concatenation of the colum vectors of the two std and the angle (shape (n, 3 . See  laueimproc.improc.spot.pca.compute_rois_pca for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_pca(  doctset: +ELLIPSIS tensor( 3.5272e+00, 2.4630e+00, -8.8861e-01], [ 2.8604e+00, 4.8286e-01, -1.5335e+00], [ 1.6986e+00, 1.5142e-01, -1.5628e+00],  ., [ 2.1297e+00, 1.7486e+00, -2.6128e-01], [ 2.0484e+00, 1.7913e+00, 4.9089e-01], [ 2.6780e+00, 1.9457e+00, 9.9400e-02 ) >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.compute_rois_sum",
"url":0,
"doc":"Sum the intensities of the pixels for each roi. Returns    - total_intensity : torch.Tensor The intensity of each roi, sum of the pixels (shape (n, . See  laueimproc.improc.spot.basic.compute_rois_sum for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_sum(  doctset: +ELLIPSIS tensor([2.1886e-01, 1.1643e-02, 5.6260e-02, 1.6938e-03, 9.8726e-03, 6.1647e-02, 3.5279e-02, 3.1891e-03, 1.0071e-02, 2.2889e-03, 1.0285e-02, 1.9760e-01, 1.7258e-02, 3.3402e-02, 9.1829e-02, 3.1510e-02, 8.4550e-02, 4.0864e-02,  ., 9.2822e-01, 9.7104e-01, 4.1733e-02, 9.4377e-02, 5.2491e-03, 4.2115e-03, 3.9217e-01, 1.5907e+00, 1.1802e+00, 1.4968e-01, 4.0696e-01, 6.3442e-01, 1.3559e+00, 6.0548e-01, 1.7116e+00, 9.2990e-01, 4.9596e-01, 2.0383e+00]) >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.fit_gaussians_em",
"url":0,
"doc":"Fit each roi by \\(K\\) gaussians using the EM algorithm. See  laueimproc.gmm for terminology, and  laueimproc.gmm.fit.fit_em for the algo description. Parameters      nbr_clusters, nbr_tries, aic, bic, log_likelihood Transmitted to  laueimproc.improc.spot.fit.fit_gaussians_em . Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). eta : torch.Tensor The relative mass \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> mean, cov, eta, _ = diagram.fit_gaussians_em(nbr_clusters=3, nbr_tries=4) >>> mean.shape, cov.shape, eta.shape (torch.Size([240, 3, 2]), torch.Size([240, 3, 2, 2]), torch.Size([240, 3] >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.fit_gaussians_mse",
"url":0,
"doc":"Fit each roi by \\(K\\) gaussians by minimising the mean square error. See  laueimproc.gmm for terminology, and  laueimproc.gmm.fit.fit_mse for the algo description. Parameters      nbr_clusters, nbr_tries, mse Transmitted to  laueimproc.improc.spot.fit.fit_gaussians_mse . Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). mag : torch.Tensor The absolute magnitude \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> mean, cov, mag, _ = diagram.fit_gaussians_mse(nbr_clusters=3, nbr_tries=4) >>> mean.shape, cov.shape, mag.shape (torch.Size([240, 3, 2]), torch.Size([240, 3, 2, 2]), torch.Size([240, 3] >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.add_property",
"url":1,
"doc":"Add a property to the diagram. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the diagram state changed.",
"func":1
},
{
"ref":"laueimproc.Diagram.areas",
"url":1,
"doc":"Return the int32 area of each bboxes. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.areas) None >>> diagram.find_spots() >>> print(diagram.areas)  doctest: +ELLIPSIS tensor([289, 36, 33, 20, 40, 81, 49, 15, 36, 25, 36, 110, 49, 56, 64, 56, 72, 56, 90, 143, 64, 42, 36, 25, 169, 110, 81, 64, 100, 49, 36, 42, 121, 36, 36, 121, 81, 56, 72, 80, 110, 56,  ., 100, 25, 225, 182, 72, 156, 90, 25, 320, 288, 144, 143, 240, 208, 64, 81, 25, 25, 144, 323, 300, 90, 144, 240, 270, 168, 352, 270, 210, 456], dtype=torch.int32) >>>"
},
{
"ref":"laueimproc.Diagram.bboxes",
"url":1,
"doc":"Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width). Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.bboxes) None >>> diagram.find_spots() >>> print(diagram.bboxes)  doctest: +ELLIPSIS tensor( 0, 0, 17, 17], [ 0, 20, 3, 12], [ 0, 1949, 3, 11],  ., [1903, 180, 18, 15], [1930, 1967, 15, 14], [1963, 1170, 24, 19 , dtype=torch.int16) >>>"
},
{
"ref":"laueimproc.Diagram.clone",
"url":1,
"doc":"Instanciate a new identical diagram. Parameters      deep : boolean, default=True If True, the memory of the new created diagram object is totally independ of this object (slow but safe). Otherwise,  self.image ,  self.rois ,  properties and  cached data share the same memory view (Tensor). So modifying one of these attributes in one diagram will modify the same attribute in the other diagram. It if realy faster but not safe. cache : boolean, default=True Copy the cache into the new diagram if True (default), or live the cache empty if False. Returns    - Diagram The new copy of self. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram_bis = diagram.clone() >>> assert id(diagram) != id(diagram_bis) >>> assert diagram.state  diagram_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.compress",
"url":1,
"doc":"Delete attributes and elements in the cache. Paremeters      size : int The quantity of bytes to remove from the cache. Returns    - removed : int The number of bytes removed from the cache.",
"func":1
},
{
"ref":"laueimproc.Diagram.file",
"url":1,
"doc":"Return the absolute file path to the image, if it is provided. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> BaseDiagram(get_sample( .file  doctest: +ELLIPSIS PosixPath('/ ./laueimproc/io/ge.jp2') >>>"
},
{
"ref":"laueimproc.Diagram.filter_spots",
"url":1,
"doc":"Keep only the given spots, delete the rest. This method can be used for filtering or sorting spots. Parameters      criteria : arraylike The list of the indices of the spots to keep (negatives indices are allow), or the boolean vector with True for keeping the spot, False otherwise like a mask. msg : str The message to happend to the history. inplace : boolean, default=True If True, modify the diagram self (no copy) and return a reference to self. If False, first clone the diagram, then apply the selection on the new diagram, It create a checkpoint (real copy) (but it is slowler). Returns    - filtered_diagram : BaseDiagram Return None if inplace is True, or a filtered clone of self otherwise. Examples     >>> from pprint import pprint >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> indices = torch.arange(0, len(diagram), 2) >>> diagram.filter_spots(indices, \"keep even spots\") >>> cond = diagram.bboxes[:, 1]  >> diag_final = diagram.filter_spots(cond, \"keep spots on left\", inplace=False) >>> pprint(diagram.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots'] >>> pprint(diag_final.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots', '120 to 63 spots: keep spots on left'] >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.find_spots",
"url":1,
"doc":"Search all the spots in this diagram, store the result into the  spots attribute. Parameters      kwargs : dict Transmitted to  laueimproc.improc.peaks_search.peaks_search .",
"func":1
},
{
"ref":"laueimproc.Diagram.get_properties",
"url":1,
"doc":"Return all the available properties.",
"func":1
},
{
"ref":"laueimproc.Diagram.get_property",
"url":1,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> diagram.add_property(\"prop2\", value=\"any python object 2\") >>> diagram.get_property(\"prop1\") 'any python object 1' >>> diagram.get_property(\"prop2\") 'any python object 2' >>> diagram.find_spots()  change state >>> diagram.get_property(\"prop1\") 'any python object 1' >>> try:  . diagram.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the diagram has changed\" >>> try:  . diagram.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.history",
"url":1,
"doc":"Return the actions performed on the Diagram since the initialisation."
},
{
"ref":"laueimproc.Diagram.is_init",
"url":1,
"doc":"Return True if the diagram has been initialized. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.is_init() False >>> diagram.find_spots() >>> diagram.is_init() True >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.image",
"url":1,
"doc":"Return the complete image of the diagram. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.image.shape torch.Size([2018, 2016]) >>> diagram.image.min() >= 0 tensor(True) >>> diagram.image.max()  >>"
},
{
"ref":"laueimproc.Diagram.plot",
"url":1,
"doc":"Prepare for display the diagram and the spots. Parameters      disp : matplotlib.figure.Figure or matplotlib.axes.Axes The matplotlib figure to complete. vmin : float, optional The minimum intensity ploted. vmax : float, optional The maximum intensity ploted. show_axis: boolean, default=True Display the label and the axis if True. show_bboxes: boolean, default=True Draw all the bounding boxes if True. show_image: boolean, default=True Display the image if True, dont call imshow otherwise. Returns    - matplotlib.axes.Axes Filled graphical widget. Notes   - It doesn't create the figure and call show. Use  self.show() to Display the diagram from scratch.",
"func":1
},
{
"ref":"laueimproc.Diagram.rawrois",
"url":1,
"doc":"Return the tensor of the raw rois of the spots."
},
{
"ref":"laueimproc.Diagram.rois",
"url":1,
"doc":"Return the tensor of the provided rois of the spots. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> diagram.rawrois.shape torch.Size([240, 24, 19]) >>> diagram.rois.shape torch.Size([240, 24, 19]) >>> diagram.rois.mean()  >>"
},
{
"ref":"laueimproc.Diagram.set_spots",
"url":1,
"doc":"Set the new spots as the current spots, reset the history and the cache. Paremeters      new_spots : tuple Can be an over diagram of type Diagram. Can be an arraylike of bounding boxes n  (anchor_i, anchor_j, height, width). Can be an anchor and a picture n  (anchor_i, anchor_j, roi). It can also be a combination of the above elements to unite the spots. Examples     >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> >>>  from diagram >>> diagram_ref = BaseDiagram(get_sample( >>> diagram_ref.find_spots() >>> diagram_ref.filter_spots(range(10 >>> >>>  from bboxes >>> bboxes =  300, 500, 10, 15], [600, 700, 20, 15 >>> >>>  from anchor and roi >>> spots =  400, 600, torch.zeros 10, 15 ], [700, 800, torch.zeros 20, 15  >>> >>>  union of the spots >>> diagram = BaseDiagram(get_sample( >>> diagram.set_spots(diagram_ref, bboxes, spots) >>> len(diagram) 14 >>>",
"func":1
},
{
"ref":"laueimproc.Diagram.state",
"url":1,
"doc":"Return a hash of the diagram. If two diagrams gots the same state, it means they are the same. The hash take in consideration the internal state of the diagram. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.state '9570d3b8743757aa3bf89a42bc4911de' >>> diagram.find_spots() >>> diagram.state '539428b799f2640150aab633de49b695' >>>"
},
{
"ref":"laueimproc.DiagramsDataset",
"url":0,
"doc":"Link Diagrams together. Initialise the dataset. Parameters       diagram_refs : tuple The diagram references, transmitted to  add_diagrams . diag2ind : callable, default=laueimproc.classes.base_dataset.default_diag2ind The function to associate an index to a diagram. If provided, this function has to be pickleable. It has to take only one positional argument (a simple Diagram instance) and to return a positive integer. diag2scalars : callable, optional If provided, it allows you to select diagram from physical parameters. \\(f:I\\to\\mathbb{R}^n\\), an injective application. \\(I\\subset\\mathbb{N}\\) is the set of the indices of the diagrams in the dataset. \\(n\\) corresponds to the number of physical parameters. For example the beam positon on the sample or, more generally, a vector of scalar parameters like time, voltage, temperature, pressure . The function must match  laueimproc.ml.dataset_dist.check_diag2scalars_typing ."
},
{
"ref":"laueimproc.DiagramsDataset.compute_mean_image",
"url":0,
"doc":"Compute the average image. Based on  laueimproc.immix.mean.mean_stack . Returns    - torch.Tensor The average image of all the images contained in this dataset.",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.compute_inter_image",
"url":0,
"doc":"Compute the median, first quartile, third quartile or everything in between. Parameters       args,  kwargs Transmitted to  laueimproc.immix.inter.sort_stack or  laueimproc.immix.inter.snowflake_stack . method : str The algorithm used. It can be  sort for the naive accurate algorithm, or  snowflake for a faster algorithm on big dataset, but less accurate. Returns    - torch.Tensor The 2d float32 grayscale image.",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.select_closest",
"url":0,
"doc":"Select the closest diagram to a given set of phisical parameters. Parameters      point, tol, scale Transmitted to  laueimproc.ml.dataset_dist.select_closest . no_raise : boolean, default = False If set to True, return None rather throwing a LookupError exception. Returns    - closet_diag :  laueimproc.classes.diagram.Diagram The closet diagram. Examples     >>> from laueimproc.classes.dataset import DiagramsDataset >>> from laueimproc.io import get_samples >>> def position(index: int) -> tuple[int, int]:  . return divmod(index, 10)  . >>> dataset = DiagramsDataset(get_samples(), diag2scalars=position) >>> dataset.select_closest 4.1, 4.9), tol=(0.2, 0.2 Diagram(img_45.jp2) >>>",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.select_closests",
"url":0,
"doc":"Select the diagrams matching a given interval of phisical parameters. Parameters      point, tol, scale Transmitted to  laueimproc.ml.dataset_dist.select_closests . Returns    - sub_dataset :  laueimproc.classes.dataset.DiagramsDataset The frozen view of this dataset containing only the diagram matching the constraints. Examples     >>> from laueimproc.classes.dataset import DiagramsDataset >>> from laueimproc.io import get_samples >>> def position(index: int) -> tuple[int, int]:  . return divmod(index, 10)  . >>> dataset = DiagramsDataset(get_samples(), diag2scalars=position) >>> dataset.select_closests 4.1, 4.9), tol=(1.2, 1.2  >>>",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.train_spot_classifier",
"url":0,
"doc":"Train a variational autoencoder classifier with the spots in the diagrams. It is a non supervised neuronal network. Parameters      model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier, optional If provided, this model will be trained again and returned. shape : float or tuple[int, int], default=0.95 The model input spot shape in numpy convention (height, width). If a percentage is given, the shape is founded with  laueimproc.nn.loader.find_shape . space : float, default = 3.0 The non penalized spreading area half size. Transmitted to  laueimproc.nn.vae_spot_classifier.VAESpotClassifier . intensity_sensitive, scale_sensitive : boolean, default=True Transmitted to  laueimproc.nn.vae_spot_classifier.VAESpotClassifier . epoch, lr, fig Transmitted to  laueimproc.nn.train.train_vae_spot_classifier . Returns    - classifier: laueimproc.nn.vae_spot_classifier.VAESpotClassifier The trained model. Examples     >>> import laueimproc >>> def init(diagram: laueimproc.Diagram):  . diagram.find_spots()  . diagram.filter_spots(range(10  to reduce the number of spots  . >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples( >>> _ = dataset.apply(init) >>> model = dataset.train_spot_classifier(epoch=2) >>>",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.add_property",
"url":2,
"doc":"Add a property to the dataset. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the dataset state changed.",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.add_diagram",
"url":2,
"doc":"Append a new diagram into the dataset. Parameters      new_diagram : laueimproc.classes.diagram.Diagram The new instanciated diagram not already present in the dataset. Raises    LookupError If the diagram is already present in the dataset.",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.add_diagrams",
"url":2,
"doc":"Append the new diagrams into the datset. Parameters      new_diagrams The diagram references, they can be of this natures:  laueimproc.classes.diagram.Diagram : Could be a simple Diagram instance.  iterable : An iterable of any of the types specified above. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> file = min(get_samples().iterdir( >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(Diagram(file  from Diagram instance >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(file)  from filename (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(file  from filename (str) >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(get_samples(  from folder (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(get_samples( )  from folder (str) >>> dataset  >>> >>>  from iterable (DiagramDataset, list, tuple,  .) >>>  ok with nested iterables >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams([Diagram(f) for f in get_samples().iterdir()]) >>> dataset  >>>",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.apply",
"url":2,
"doc":"Apply an operation in all the diagrams of the dataset. Parameters      func : callable A function that take a diagram and optionaly other parameters, and return anything. The function can modify the diagram inplace. It has to be pickaleable. args : tuple, optional Positional arguments transmitted to the provided func. kwargs : dict, optional Keyword arguments transmitted to the provided func. Returns    - res : dict The result of the function for each diagrams. To each diagram index, associate the result. Notes   - This function will be automaticaly applided on the new diagrams, but the result is throw. Examples     >>> from pprint import pprint >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( [ 10]  subset to go faster >>> def peak_search(diagram: Diagram, density: float) -> int:  .  'Return the number of spots. '  . diagram.find_spots(density=density)  . return len(diagram)  . >>> res = dataset.apply(peak_search, args=(0.5, >>> pprint(res) {0: 204, 10: 547, 20: 551, 30: 477, 40: 404, 50: 537, 60: 2121, 70: 274, 80: 271, 90: 481} >>>",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.autosave",
"url":2,
"doc":"Manage the dataset recovery. This allows the dataset to be backuped at regular time intervals. Parameters      filename : pathlike The name of the persistant file, transmitted to  laueimproc.io.save_dataset.write_dataset . delay : float or str, optional If provided and > 0, automatic checkpoint will be performed. it corresponds to the time interval between two check points. The supported formats are defined in  laueimproc.common.time2sec .",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.clone",
"url":2,
"doc":"Instanciate a new identical dataset. Parameters       kwargs : dict Transmitted to  laueimproc.classes.base_diagram.BaseDiagram.clone . Returns    - BaseDiagramsDataset The new copy of self. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset_bis = dataset.clone() >>> assert id(dataset) != id(dataset_bis) >>> assert dataset.state  dataset_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.flush",
"url":2,
"doc":"Extract finished thread diagrams.",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.get_property",
"url":2,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> dataset.add_property(\"prop2\", value=\"any python object 2\") >>> dataset.get_property(\"prop1\") 'any python object 1' >>> dataset.get_property(\"prop2\") 'any python object 2' >>> dataset = dataset[:1]  change state >>> dataset.get_property(\"prop1\") 'any python object 1' >>> try:  . dataset.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the dataset has changed\" >>> try:  . dataset.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.get_scalars",
"url":2,
"doc":"Return the scalars values of each diagrams given by  diag2scalars .",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.indices",
"url":2,
"doc":"Return the sorted map diagram indices currently reachable in the dataset. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset[-10:10:-5].indices [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90] >>>"
},
{
"ref":"laueimproc.DiagramsDataset.state",
"url":2,
"doc":"Return a hash of the dataset. If two datasets gots the same state, it means they are the same. The hash take in consideration the indices of the diagrams and the functions applyed. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:].state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:1].state '1de1605a297bafd22a886de7058cae81' >>>"
},
{
"ref":"laueimproc.DiagramsDataset.restore",
"url":2,
"doc":"Restore the dataset content from the backup file. Do nothing if the file doesn't exists. Based on  laueimproc.io.save_dataset.restore_dataset . Parameters      filename : pathlike The name of the persistant file.",
"func":1
},
{
"ref":"laueimproc.DiagramsDataset.run",
"url":2,
"doc":"Run asynchronousely in a child thread, called by self.start().",
"func":1
},
{
"ref":"laueimproc.io",
"url":3,
"doc":"Read an write the data on the harddisk, persistant layer."
},
{
"ref":"laueimproc.io.get_sample",
"url":3,
"doc":"Return the path of the test image. Examples     >>> from laueimproc.io import get_sample >>> get_sample().name 'ge.jp2' >>> get_sample().exists() True >>>",
"func":1
},
{
"ref":"laueimproc.io.get_samples",
"url":3,
"doc":"Download the samples of laue diagram and return the folder. Parameters       kwargs : dict Transmitted to  download . Returns    - folder : pathlib.Path The folder containing all the samples.",
"func":1
},
{
"ref":"laueimproc.io.write_dat",
"url":4,
"doc":"Write a dat file."
},
{
"ref":"laueimproc.io.write_dat.write_dat",
"url":4,
"doc":"Write the dat file associate to the provided diagram. The file contains the following columns:  peak_X : float The x position of the center of the peak in convention j+1/2 in this module. It is like cv2 convention but start by (1, 1), not (0, 0). The position is the position estimated after a gaussian fit with mse criteria.  peak_Y : float The y position of the center of the peak in convention i+1/2 in this module. Same formalism as peak_X.  peak_Itot : float peak_Isub + the mean of the background estimation in the roi.  peak_Isub : float Gaussian amplitude, fitted on the roi with no background. Intensity without background.  peak_fwaxmaj : float Main full with. It is 2 std of the gaussian in the main axis of the PCA.  peak_fwaxmin : float Same as peak_fwaxmaj along the secondary axis.  peak_inclination : float The angle between the vertical axis (i) and the main pca axis of the spot. The angle is defined in clockwise (-trigo_angle(i, j in order to give a positive angle when we plot the image and when we have a view in the physician base (-j, i). The angle is defined between -90 and 90 degrees.  Xdev : float The difference between the position of the gaussian and the baricenter along the x axis in cv2 convention (axis j in this module).  Ydev : float Same as Xdev for the y axis.  peak_bkg : float The mean of the estimated background in the roi.  Ipixmax : int The max intensity of the all the pixel in the roi with no background. The peaks are written in the native order of the gaussian. Parameters      filename : pathlike The path to the file, relative or absolute. The suffix \".dat\" is automaticaly append if it is not already provided. If a file is already existing, it is overwriten. diagram : laueimproc.classes.diagram.Diagram The reference to the diagram. Examples     >>> import pathlib >>> import tempfile >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io.write_dat import write_dat >>> import laueimproc.io >>> diagram = Diagram(pathlib.Path(laueimproc.io.__file__).parent / \"ge.jp2\") >>> diagram.find_spots() >>> file = pathlib.Path(tempfile.gettempdir( / \"ge_blanc.dat\" >>> write_dat(file, diagram) >>> >>> with open(file, \"r\", encoding=\"utf-8\") as raw:  doctest: +ELLIPSIS  . print(raw.read(  . peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj  .ation Xdev Ydev peak_bkg Ipixmax 5.549 4.764 1500.0 604.0 7.054  .-50.9 0.000 0.000 896.0 1121.0 26.591 1.228 1116.4 89.0 5.721  .-87.9 0.000 0.000 1027.4 1129.0  . 1974.962 1938.139 4998.9 3950.0 4.097  . 28.1 0.000 0.000 1048.9 4999.0 1179.912 1976.142 20253.1 19144.0 5.356  . 5.7 0.000 0.000 1109.1 20258.0  file created by laueimproc  .  Diagram from ge.jp2:  History:  1. 240 spots from self.find_spots()  No Properties  Current state:   id, state:  .   nbr spots: 240   total mem: 16.4MB >>>",
"func":1
},
{
"ref":"laueimproc.io.write",
"url":5,
"doc":"Write an image of laue diagram."
},
{
"ref":"laueimproc.io.write.write_jp2",
"url":5,
"doc":"Write a lossless jpeg2000 image with the metadata. Parameters      filename : pathlike The path to the file, relative or absolute. The suffix \".jp2\" is automaticaly append if it is not already provided. If a file is already existing, it is overwriten. image : torch.Tensor The 2d grayscale float image with values in range [0, 1]. metadata : dict, optional If provided, a jsonisable dictionary of informations. Examples     >>> import pathlib >>> import tempfile >>> import torch >>> from laueimproc.io.read import read_image >>> from laueimproc.io.write import write_jp2 >>> >>> file = pathlib.Path(tempfile.gettempdir( / \"image.jp2\" >>> img_ref = torch.rand 2048, 1024 >>> img_ref = (img_ref  65535 + 0.5).to(int).to(torch.float32) / 65535 >>> metadata_ref = {\"prop1\": 1} >>> >>> write_jp2(file, img_ref, metadata_ref) >>> img, metadata = read_image(file) >>> torch.allclose(img, img_ref, atol=1/65535, rtol=0) True >>> metadata  metadata_ref True >>>",
"func":1
},
{
"ref":"laueimproc.io.save_dataset",
"url":6,
"doc":"Write an load a dataset."
},
{
"ref":"laueimproc.io.save_dataset.restore_dataset",
"url":6,
"doc":"Load or update the content of the dataset. Parameters      filename : pathlib.Path The filename of the pickle file, including the extension. dataset : laueimproc.classes.base_dataset.BaseDiagramsDataset, optional If provided, update the state of the dataset by the recorded content. If it is not provided, a new dataset is created. Returns    - dataset : laueimproc.classes.base_dataset.BaseDiagramsDataset A reference to the updated dataset. Examples     >>> import pathlib >>> import tempfile >>> from laueimproc.io.save_dataset import restore_dataset, write_dataset >>> import laueimproc >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples( >>> dataset.state '047e5d6c00850898c233128e31e1f7e1' >>> file = pathlib.Path(tempfile.gettempdir( / \"dataset.pickle\" >>> file = write_dataset(file, dataset) >>> empty_dataset = laueimproc.DiagramsDataset() >>> empty_dataset.state '746e927ad8595aa230adef17f990ba96' >>> filled_dataset = restore_dataset(file, empty_dataset) >>> filled_dataset.state '047e5d6c00850898c233128e31e1f7e1' >>> empty_dataset is filled_dataset  update state inplace True >>>",
"func":1
},
{
"ref":"laueimproc.io.save_dataset.write_dataset",
"url":6,
"doc":"Write the pickle file associate to the provided diagrams dataset. Parameters      filename : pathlike The path to the file, relative or absolute. The suffix \".pickle\" is automaticaly append if it is not already provided. If a file is already existing, it is overwriten. dataset : laueimproc.classes.base_dataset.BaseDiagramsDataset The reference to the diagrams dataset. Returns    - filename : pathlib.Path The real absolute filename used.",
"func":1
},
{
"ref":"laueimproc.io.read",
"url":7,
"doc":"Read an image of laue diagram."
},
{
"ref":"laueimproc.io.read.extract_metadata",
"url":7,
"doc":"Extract the metadata of the file content. Parameters      data : bytes The raw image file content. Returns    - metadata : dict The metadata of the file.",
"func":1
},
{
"ref":"laueimproc.io.read.read_image",
"url":7,
"doc":"Read and decode a grayscale image into a numpy array. Use cv2 as possible, and fabio if cv2 failed. Parameters      filename : pathlike The path to the image, relative or absolute. Returns    - image : torch.Tensor The grayscale laue image matrix in float between with value range in [0, 1]. metadata : dict The file metadata. Raises    OSError If the given path is not a file, or if the image reading failed.",
"func":1
},
{
"ref":"laueimproc.io.read.to_floattensor",
"url":7,
"doc":"Convert and shift tenso into float torch tensor. If the input is not in floating point, it is converting in float32 and the value range is set beetweeen 0 and 1. Parameters      data : arraylike The torch tensor or the numpy array to convert. Returns    - tensor : torch.Tensor The float torch tensor.",
"func":1
},
{
"ref":"laueimproc.io.download",
"url":8,
"doc":"Download dataset or models from internet."
},
{
"ref":"laueimproc.io.download.download",
"url":8,
"doc":"Download, decompress and unpack the data given by the url. Parameters      url : str The internet link to download the file directely. folder : pathlike, optional The folder to store the data. force_download : boolean, default=False If set to True, download even if the data are stored localy. Returns    - pathlib.Path The path of the final data.",
"func":1
},
{
"ref":"laueimproc.io.download.get_samples",
"url":8,
"doc":"Download the samples of laue diagram and return the folder. Parameters       kwargs : dict Transmitted to  download . Returns    - folder : pathlib.Path The folder containing all the samples.",
"func":1
},
{
"ref":"laueimproc.io.convert",
"url":9,
"doc":"Convert the image file format."
},
{
"ref":"laueimproc.io.convert.converter_decorator",
"url":9,
"doc":"Decorate an image converter into a batch image converter.",
"func":1
},
{
"ref":"laueimproc.io.convert.to_jp2",
"url":9,
"doc":"Convert a file into jpeg2000 format. Parameters      src_file : pathlibPath The input filename. dst_dir : pathlib.Path The output directory. metadata : boolean Flag to allow the copy of metadata. Returns    - abs_out_path : pathlib.Path The absolute filename of the created image.",
"func":1
},
{
"ref":"laueimproc.io.comp_lm",
"url":10,
"doc":"Implement a convolutive generative variational auto-encoder neuronal network."
},
{
"ref":"laueimproc.io.comp_lm.VariationalEncoder",
"url":10,
"doc":"Projects images into a more compact space. Each patch of 320x320 pixels with a stride of 64 pixels is projected into a space of dimension 64, quantizable with 8 bits per component. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"laueimproc.io.comp_lm.VariationalEncoder.forward",
"url":10,
"doc":"Apply the function on the images. Parameters      img : torch.Tensor The float image batch of shape (n, 1, h, w). With h and w >= 160 + k 32, k >= 0 integer. Returns    - lat : torch.Tensor The projection of the image in the latent space. New shape is (n, 256, h/32-4, w/32-4) with value in [0, 1]. Examples     >>> import torch >>> from laueimproc.io.comp_lm import VariationalEncoder >>> encoder = VariationalEncoder() >>> encoder(torch.rand 5, 1, 160, 224 ).shape torch.Size([5, 64, 1, 3]) >>>",
"func":1
},
{
"ref":"laueimproc.io.comp_lm.VariationalEncoder.add_quantization_noise",
"url":10,
"doc":"Add a uniform noise in order to simulate the quantization into uint8. Parameters      lat : torch.Tensor The float lattent space of shape (n, 64, a, b) with value in range ]0, 1[. Returns    - noised_lat : torch.Tensor The input tensor with a aditive uniform noise U(-.5/255, .5/255). The finals values are clamped to stay in the range [0, 1]. Examples     >>> import torch >>> from laueimproc.io.comp_lm import VariationalEncoder >>> lat = torch.rand 5, 64, 1, 3 >>> q_lat = VariationalEncoder.add_quantization_noise(lat) >>> torch.all(abs(q_lat - lat)  >> abs q_lat - lat).mean().round(decimals=3 tensor(0.) >>>",
"func":1
},
{
"ref":"laueimproc.io.comp_lm.Decoder",
"url":10,
"doc":"Unfold the projected encoded images into the color space. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"laueimproc.io.comp_lm.Decoder.forward",
"url":10,
"doc":"Apply the function on the latent images. Parameters      lat : torch.Tensor The projected image in the latent space of shape (n, 256, hl, wl). Returns    - img : torch.Tensor A close image in colorspace to the input image. It is as mutch bijective as possible than VariationalEncoder. New shape is (n, 256, 160+hl 32, 160+wl 32) with value in [0, 1]. Examples     >>> import torch >>> from laueimproc.io.comp_lm import Decoder >>> decoder = Decoder() >>> decoder(torch.rand 5, 64, 1, 3 ).shape torch.Size([5, 1, 160, 224]) >>>",
"func":1
},
{
"ref":"laueimproc.io.comp_lm.LMCodec",
"url":10,
"doc":"Encode and Decode Laue Max images. Initialise the codec. Parameters      weights : pathlike The filename of the model data."
},
{
"ref":"laueimproc.io.comp_lm.LMCodec.decode",
"url":10,
"doc":"Decode le compressed content. Parameters      data The compact data representation. Examples     >>> import numpy as np >>> from laueimproc.io.comp_lm import LMCodec >>> codec = LMCodec(\"/tmp/lmweights.tar\").eval() >>> img = np.random.randint(0, 65536, (1000, 1000), dtype=np.uint16) >>> data = codec.encode(img) >>> decoded = codec.decode(data) >>> bool img  decoded).all( True >>>",
"func":1
},
{
"ref":"laueimproc.io.comp_lm.LMCodec.encode",
"url":10,
"doc":"Encode the image. Parameters      img : torch.Tensor or np.ndarray The 1 channel image to encode. Examples     >>> import numpy as np >>> from laueimproc.io.comp_lm import LMCodec >>> codec = LMCodec(\"/tmp/lmweights.tar\").eval() >>> img = np.random.randint(0, 65536, (1000, 1000), dtype=np.uint16) >>> encoded = codec.encode(img) >>>",
"func":1
},
{
"ref":"laueimproc.io.comp_lm.LMCodec.forward",
"url":10,
"doc":"Encode then decode the image. Parameters      img : torch.Tensor The 1 channel image to encode. Returns    - predicted : torch.Tensor The predicted image of same shape. Examples     >>> import torch >>> from laueimproc.io.comp_lm import LMCodec >>> codec = LMCodec(\"/tmp/lmweights.tar\") >>> codec.forward(torch.rand 1000, 1000 ).shape torch.Size([1000, 1000]) >>>",
"func":1
},
{
"ref":"laueimproc.io.comp_lm.LMCodec.overfit",
"url":10,
"doc":"Overfit the model on the given diagrams. Parameters      diagrams : list[Diagram] All the diagrams we want to compress.",
"func":1
},
{
"ref":"laueimproc.improc",
"url":11,
"doc":"Tensor processing for Laue."
},
{
"ref":"laueimproc.improc.spot",
"url":12,
"doc":"Implement image processing tools to annalyse the spots."
},
{
"ref":"laueimproc.improc.spot.basic",
"url":13,
"doc":"Give simple information on spots."
},
{
"ref":"laueimproc.improc.spot.basic.compute_rois_centroid",
"url":13,
"doc":"Find the weighted barycenter of each roi. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - positions : torch.Tensor The position of the height and width barycenter for each roi of shape (n, 2). Examples     >>> import torch >>> from laueimproc.improc.spot.basic import compute_rois_centroid >>> patches = [  . torch.tensor( 0.5 ),  . torch.tensor( 0.5, 0.5], [0.5, 1.0 ),  . torch.tensor( 0.0, 0.0, 0.5, 0.0, 0.0 ),  . ]  2 >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes( >>> bboxes = torch.tensor( 0, 0, 1, 1],  . [ 0, 0, 2, 2],  . [ 0, 0, 1, 5],  . [10, 10, 1, 1],  . [10, 10, 2, 2],  . [10, 10, 1, 5 , dtype=torch.int16) >>> print(compute_rois_centroid(data, bboxes tensor( 0.5000, 0.5000], [ 1.1000, 1.1000], [ 0.5000, 2.5000], [10.5000, 10.5000], [11.1000, 11.1000], [10.5000, 12.5000 ) >>> torch.allclose(  . compute_rois_centroid(data, bboxes),  . compute_rois_centroid(data, bboxes, _no_c=True),  . ) True >>>",
"func":1
},
{
"ref":"laueimproc.improc.spot.basic.compute_rois_inertia",
"url":13,
"doc":"Return the order 2 momentum centered along the barycenter. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - inertia : torch.Tensor The order 2 momentum centered along the barycenter (shape (n, . Examples     >>> import torch >>> from laueimproc.improc.spot.basic import compute_rois_inertia >>> patches = [  . torch.tensor( 0.5 ),  . torch.tensor( 1.0 ),  . torch.tensor( 0.5, 0.5], [0.5, 0.5 ),  . torch.tensor( 0.2, 0.0, 1.0, 0.0, 0.2 ),  . ] >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes( >>> bboxes = torch.tensor( 0, 0, 1, 1],  . [ 0, 0, 1, 1],  . [ 0, 0, 2, 2],  . [ 0, 0, 1, 5 , dtype=torch.int16) >>> print(compute_rois_inertia(data, bboxes tensor([0.0000, 0.0000, 1.0000, 1.6000]) >>> torch.allclose(  . compute_rois_inertia(data, bboxes),  . compute_rois_inertia(data, bboxes, _no_c=True),  . ) True >>>",
"func":1
},
{
"ref":"laueimproc.improc.spot.basic.compute_rois_max",
"url":13,
"doc":"Return the argmax of the intensity and the intensity max of each roi. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - pos1_pos2_imax : torch.Tensor The concatenation of the colum vectors of the argmax along i axis, argmax along j axis and the max intensity. The shape is (n, 3). Examples     >>> import torch >>> from laueimproc.improc.spot.basic import compute_rois_max >>> patches = [  . torch.tensor( 0.5 ),  . torch.tensor( 0.5, 0.5], [0.5, 1.0 ),  . torch.tensor( 0.0, 0.0, 0.5, 0.0, 0.0 ),  . ]  2 >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes( >>> bboxes = torch.tensor( 0, 0, 1, 1],  . [ 0, 0, 2, 2],  . [ 0, 0, 1, 5],  . [10, 10, 1, 1],  . [10, 10, 2, 2],  . [10, 10, 1, 5 , dtype=torch.int16) >>> print(compute_rois_max(data, bboxes tensor( 0.5000, 0.5000, 0.5000], [ 1.5000, 1.5000, 1.0000], [ 0.5000, 2.5000, 0.5000], [10.5000, 10.5000, 0.5000], [11.5000, 11.5000, 1.0000], [10.5000, 12.5000, 0.5000 ) >>> torch.allclose(  . compute_rois_max(data, bboxes),  . compute_rois_max(data, bboxes, _no_c=True),  . ) True >>>",
"func":1
},
{
"ref":"laueimproc.improc.spot.basic.compute_rois_sum",
"url":13,
"doc":"Return the intensity of each roi, sum of the pixels. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - intensity : torch.Tensor The vector of the intensity of shape (n,). Examples     >>> import torch >>> from laueimproc.improc.spot.basic import compute_rois_sum >>> patches = [  . torch.tensor( 0.5 ),  . torch.tensor( 0.5, 0.5], [0.5, 1.0 ),  . torch.tensor( 0.0, 0.0, 0.5, 0.0, 0.0 ),  . ] >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes( >>> bboxes = torch.tensor( 0, 0, 1, 1],  . [ 0, 0, 2, 2],  . [ 0, 0, 1, 5 , dtype=torch.int16) >>> print(compute_rois_sum(data, bboxes tensor([0.5000, 2.5000, 0.5000]) >>> torch.allclose(  . compute_rois_sum(data, bboxes),  . compute_rois_sum(data, bboxes, _no_c=True),  . ) True >>>",
"func":1
},
{
"ref":"laueimproc.improc.spot.pca",
"url":14,
"doc":"Efficient Principal Componant Annalysis on spots."
},
{
"ref":"laueimproc.improc.spot.pca.compute_rois_pca",
"url":14,
"doc":"Compute the PCA for each spot. See  laueimproc.gmm for terminology. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - pca : torch.Tensor The concatenation of the colum vectors of the std in pixel and the angle in radian \\( \\left[ \\sqrt{\\lambda_1}, \\sqrt{\\lambda_2}, \\theta \\right] \\) of shape (n, 3). The diagonalization of \\(\\mathbf{\\Sigma}\\) is performed with  laueimproc.gmm.linalg.cov2d_to_eigtheta . Examples     >>> from math import cos, sin >>> import numpy as np >>> import torch >>> from laueimproc.improc.spot.pca import compute_rois_pca >>> >>> lambda1, lambda2, theta = 100 2, 50 2, 0.3 >>> rot = torch.asarray( cos(theta), -sin(theta)], [sin(theta), cos(theta) ) >>> diag = torch.asarray( lambda1, 0.0], [0.0, lambda2 ) >>> cov = rot @ diag @ rot.mT >>> cov = 0.5  (cov + cov.mT)  solve warn covariance is not symmetric positive-semidefinite >>> np.random.seed(0) >>> points = torch.from_numpy(np.random.multivariate_normal([0, 0], cov, 1_000_000 .to(int) >>> points -= points.amin(dim=0).unsqueeze(0) >>> height, width = int(points[:, 0].max() + 1), int(points[:, 1].max() + 1) >>> rois = points[:, 0] width + points[:, 1] >>> rois = torch.bincount(rois, minlength=height width).to(torch.float32) >>> rois /= rois.max()  in [0, 1] >>> rois = rois.reshape(1, height, width) >>> compute_rois_pca(  . bytearray(rois.numpy().tobytes( ,  . torch.asarray( 0, 0, height, width , dtype=torch.int16)  . ) tensor( 99.4283, 49.6513, 0.2973 ) >>> >>> rois = np.zeros 4, 5, 5), np.float32) >>> rois[0, :, 2] = 1 >>> rois[1, range(5), range(5)] = 1 >>> rois[2, 2, :] = 1 >>> rois[3, :, 2] = 1 >>> rois[3, 2, 1:4] = 1 >>> rois array( [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0. ,   1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1. ,   0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0. ,   0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 1., 1., 1., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0. ], dtype=float32) >>> data = bytearray(rois.tobytes( >>> bboxes = torch.asarray( 0, 0, 5, 5  4, dtype=torch.int16) >>> compute_rois_pca(data, bboxes) tensor( 1.4142, 0.0000, 0.0000], [2.0000, 0.0000, 0.7854], [1.4142, 0.0000, 1.5708], [1.1952, 0.5345, 0.0000 ) >>> >>> torch.allclose(_, compute_rois_pca(data, bboxes, _no_c=True True >>>",
"func":1
},
{
"ref":"laueimproc.improc.spot.fit",
"url":15,
"doc":"Fit the spot by one gaussian."
},
{
"ref":"laueimproc.improc.spot.fit.fit_gaussians_em",
"url":15,
"doc":"Fit each roi by \\(K\\) gaussians. See  laueimproc.gmm for terminology. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. nbr_clusters, nbr_tries : int Transmitted to  laueimproc.gmm.fit.fit_em . aic, bic, log_likelihood : boolean, default=False If set to True, the metric is happend into  infodict . The metrics are computed in the  laueimproc.gmm.metric module.  aic: Akaike Information Criterion of shape ( .,). \\(aic = 2p-2\\log(L_{\\alpha,\\omega})\\), \\(p\\) is the number of free parameters and \\(L_{\\alpha,\\omega}\\) the likelihood.  bic: Bayesian Information Criterion of shape ( .,). \\(bic = \\log(N)p-2\\log(L_{\\alpha,\\omega})\\), \\(p\\) is the number of free parameters and \\(L_{\\alpha,\\omega}\\) the likelihood.  log_likelihood of shape ( .,): \\( \\log\\left(L_{\\alpha,\\omega}\\right) = \\log\\left( \\prod\\limits_{i=1}^N \\sum\\limits_{j=1}^K \\eta_j \\left( \\mathcal{N}_{(\\mathbf{\\mu}_j,\\frac{1}{\\omega_i}\\mathbf{\\Sigma}_j)}(\\mathbf{x}_i) \\right)^{\\alpha_i} \\right) \\) Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). eta : torch.Tensor The relative mass \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> import torch >>> from laueimproc.gmm.linalg import multivariate_normal >>> from laueimproc.improc.spot.fit import fit_gaussians_em >>> >>> cov_ref = torch.asarray( [2, 1], [1, 4 ,  3, -3], [-3, 9 ,  4, 0], [0, 4 ]) >>> cov_ref = cov_ref.to(torch.float32) >>> mean_ref = torch.asarray( [10.0, 15.0], [20.0, 15.0], [15.0, 20.0 ]) >>> obs = multivariate_normal(cov_ref, 3_000_000) + mean_ref >>> obs = obs.to(torch.int32).reshape(-1, 2) >>> height, width = 35, 35 >>> rois = obs[:, 0] width + obs[:, 1] >>> rois = rois[torch.logical_and(rois >= 0, rois  >> rois = torch.bincount(rois, minlength=height width).to(torch.float32) >>> rois = rois.reshape(1, height, width) / rois.max() >>> >>> data = bytearray(rois.to(torch.float32).numpy().tobytes( >>> bboxes = torch.asarray( 0, 0, height, width , dtype=torch.int16) >>> mean, cov, eta, infodict = fit_gaussians_em(  . data, bboxes, nbr_clusters=3, aic=True, bic=True, log_likelihood=True  . ) >>> any(torch.allclose(mu, mean_ref[0, 0], atol=0.5) for mu in mean[0]) True >>> any(torch.allclose(mu, mean_ref[0, 1], atol=0.5) for mu in mean[0]) True >>> any(torch.allclose(mu, mean_ref[0, 2], atol=0.5) for mu in mean[0]) True >>> any(torch.allclose(c, cov_ref[0], atol=1.0) for c in cov[0]) True >>> any(torch.allclose(c, cov_ref[1], atol=1.0) for c in cov[0]) True >>> any(torch.allclose(c, cov_ref[2], atol=1.0) for c in cov[0]) True >>> >>>  import matplotlib.pyplot as plt >>>  _ = plt.imshow(rois[0], extent=(0, width, height, 0 >>>  _ = plt.scatter(mean[ ., 1].ravel(), mean[ ., 0].ravel( >>>  plt.show() >>>",
"func":1
},
{
"ref":"laueimproc.improc.spot.fit.fit_gaussians_mse",
"url":15,
"doc":"Fit each roi by \\(K\\) gaussians by minimising the mean square error. See  laueimproc.gmm for terminology. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. nbr_clusters, nbr_tries : int Transmitted to  laueimproc.gmm.fit.fit_mse . mse : boolean, default=False If set to True, the metric is happend into  infodict . The metrics are computed in the  laueimproc.gmm.metric module.  mse: Mean Square Error of shape ( .,). \\( mse = \\frac{1}{N}\\sum\\limits_{i=1}^N \\left(\\left(\\sum\\limits_{j=1}^K \\eta_j \\left( \\mathcal{N}_{(\\mathbf{\\mu}_j,\\frac{1}{\\omega_i}\\mathbf{\\Sigma}_j)}(\\mathbf{x}_i) \\right)\\right) - \\alpha_i\\right)^2 \\) Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). mag : torch.Tensor The absolute magnitude \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> import torch >>> from laueimproc.gmm.linalg import multivariate_normal >>> from laueimproc.improc.spot.fit import fit_gaussians_mse >>> >>> cov_ref = torch.asarray( [2, 1], [1, 4 ,  3, -3], [-3, 9 ]) >>> cov_ref = cov_ref.to(torch.float32) >>> mean_ref = torch.asarray( [10.0, 15.0], [20.0, 15.0 ]) >>> obs = multivariate_normal(cov_ref, 2_000_000) + mean_ref >>> obs = obs.to(torch.int32).reshape(-1, 2) >>> height, width = 30, 30 >>> rois = obs[:, 0] width + obs[:, 1] >>> rois = rois[torch.logical_and(rois >= 0, rois  >> rois = torch.bincount(rois, minlength=height width).to(torch.float32) >>> rois = rois.reshape(1, height, width) / rois.max() >>> >>> data = bytearray(rois.to(torch.float32).numpy().tobytes( >>> bboxes = torch.asarray( 0, 0, height, width , dtype=torch.int16) >>> mean, cov, mag, _ = fit_gaussians_mse(data, bboxes, nbr_clusters=2) >>> any(torch.allclose(mu, mean_ref[0, 0], atol=0.5) for mu in mean[0]) True >>> any(torch.allclose(mu, mean_ref[0, 1], atol=0.5) for mu in mean[0]) True >>> >>>  import matplotlib.pyplot as plt >>>  _ = plt.imshow(rois[0], extent=(0, width, height, 0 >>>  _ = plt.scatter(mean[ ., 1].ravel(), mean[ ., 0].ravel( >>>  plt.show() >>>",
"func":1
},
{
"ref":"laueimproc.improc.spot.extrema",
"url":16,
"doc":"Search the number of extremums in each rois."
},
{
"ref":"laueimproc.improc.spot.extrema.compute_rois_nb_peaks",
"url":16,
"doc":"Find the number of extremums in each roi. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - nbr : torch.Tensor The number of extremum in each roi of shape (n,) and dtype int16. The detection is based on canny filtering. Examples     >>> import torch >>> from laueimproc.improc.spot.extrema import compute_rois_nb_peaks >>> patches = [  . torch.tensor( .5 ),  . torch.tensor( .5, .5], [.5, .5 ),  . torch.tensor( .5, .5, .5], [.5, .5, .5], [.5, .5, .5 ),  . torch.tensor( .1, .0, .1, .2, .3, .3, .2, .1, .1],  . [.0, .2, .1, .2, .3, .2, .1, .1, .1],  . [.0, .1, .2, .2, .2, .1, .1, .0, .0],  . [.1, .0, .1, .1, .2, .1, .0, .0, .0],  . [.2, .1, .2, .3, .4, .6, 4., .3, .1],  . [.3, .2, .3, .5, .8, .8, .8, .6, .3],  . [.4, .3, .5, .8, .8, .8, .8, .7, .3],  . [.2, .3, .5, .7, .8, .8, .7, .6, .3],  . [.2, .2, .4, .6, .7, .7, .5, .3, .2 ),  . ] >>> data = bytearray(torch.cat([p.ravel() for p in patches]).numpy().tobytes( >>> bboxes = torch.tensor( 0, 0, 1, 1],  . [ 0, 0, 2, 2],  . [ 0, 0, 3, 3],  . [ 0, 0, 9, 9 , dtype=torch.int16) >>> print(compute_rois_nb_peaks(data, bboxes tensor([1, 1, 1, 5], dtype=torch.int16) >>> (  . compute_rois_nb_peaks(data, bboxes)  compute_rois_nb_peaks(data, bboxes, _no_c=True)  . ).all() tensor(True) >>>",
"func":1
},
{
"ref":"laueimproc.improc.morpho",
"url":17,
"doc":"Morphological image operation with rond kernels."
},
{
"ref":"laueimproc.improc.morpho.get_circle_kernel",
"url":17,
"doc":"Compute a circle structurant element. Parameters      radius : float The radius of the circle, such as diameter is 2 radius. Return    circle : torch.Tensor The float32 2d image of the circle. The dimension size dim is odd. Notes   - Not equivalent to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 radius, 2 radius . Examples     >>> from laueimproc.improc.morpho import get_circle_kernel >>> get_circle_kernel(0) tensor( 1 , dtype=torch.uint8) >>> get_circle_kernel(1) tensor( 0, 1, 0], [1, 1, 1], [0, 1, 0 , dtype=torch.uint8) >>> get_circle_kernel(2) tensor( 0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0 , dtype=torch.uint8) >>> get_circle_kernel(3) tensor( 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0 , dtype=torch.uint8) >>> get_circle_kernel(6) tensor( 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , dtype=torch.uint8) >>> get_circle_kernel(2.5) tensor( 0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0 , dtype=torch.uint8) >>>",
"func":1
},
{
"ref":"laueimproc.improc.morpho.morphology",
"url":17,
"doc":"Apply the morphological operation on the image. Parameters      image : torch.Tensor The float32 c contiguous 2d image. radius : float The radius of the structurant rond kernel. operation : str The operation type, can be \"close\", \"dilate\", \"erode\", \"open\". Returns    - image : torch.Tensor The transformed image as a reference to input image. Examples     >>> import torch >>> from laueimproc.improc.morpho import morphology >>> image = torch.rand 2048, 2048 >>> out = morphology(image.clone(), radius=3, operation=\"open\") >>> out.sum()  >> out = morphology(image.clone(), radius=3, operation=\"close\") >>> out.sum() > image.sum() tensor(True) >>>",
"func":1
},
{
"ref":"laueimproc.improc.morpho.morpho_close",
"url":17,
"doc":"Apply a morphological closing on the image. Parameters      image : torch.Tensor The float32 c contiguous 2d image. radius : float The radius of the structurant rond kernel. Returns    - image : torch.Tensor The closed image as a reference to the input image.",
"func":1
},
{
"ref":"laueimproc.improc.morpho.morpho_dilate",
"url":17,
"doc":"Apply a morphological dilatation on the image. Parameters      image : torch.Tensor The float32 c contiguous 2d image. radius : float The radius of the structurant rond kernel. Returns    - image : torch.Tensor The dilated image as a reference to the input image. Examples     >>> import torch >>> from laueimproc.improc.morpho import morpho_dilate >>> image = torch.tensor( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  . [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],  . [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [0, 0, 1, 0, 0, 0, 0, 0, 0, 1 , dtype=torch.float32) >>> morpho_dilate(image, radius=2) tensor( 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.], [0., 1., 1., 1., 0., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [0., 1., 1., 1., 0., 0., 1., 1., 1., 1.], [0., 0., 1., 0., 0., 0., 0., 1., 1., 0.], [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 0., 0., 0., 0., 0., 0., 1.], [1., 1., 1., 1., 0., 0., 0., 0., 1., 1.], [1., 1., 1., 1., 1., 0., 0., 1., 1., 1. ) >>>",
"func":1
},
{
"ref":"laueimproc.improc.morpho.morpho_erode",
"url":17,
"doc":"Apply a morphological erosion on the image. Parameters      image : torch.Tensor The float32 c contiguous 2d image. radius : float The radius of the structurant rond kernel. Returns    - image : torch.Tensor The eroded image as a reference to the input image. Examples     >>> import torch >>> from laueimproc.improc.morpho import morpho_erode >>> image = torch.tensor( 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],  . [0, 1, 1, 1, 0, 1, 1, 1, 1, 1],  . [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  . [0, 1, 1, 1, 0, 0, 1, 1, 1, 1],  . [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],  . [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  . [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],  . [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],  . [1, 1, 1, 1, 1, 0, 0, 1, 1, 1 , dtype=torch.float32) >>> morpho_erode(image, radius=2) tensor( 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.], [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.], [0., 0., 1., 0., 0., 0., 0., 1., 1., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 0., 0., 0., 0., 0., 0., 1. ) >>>",
"func":1
},
{
"ref":"laueimproc.improc.morpho.morpho_open",
"url":17,
"doc":"Apply a morphological opening on the image. Parameters      image : torch.Tensor The float32 c contiguous 2d image. radius : float The radius of the structurant rond kernel. Returns    - image : torch.Tensor The opend image as a reference to the input image.",
"func":1
},
{
"ref":"laueimproc.improc.find_bboxes",
"url":18,
"doc":"Find the bboxes of a binary image."
},
{
"ref":"laueimproc.improc.find_bboxes.find_bboxes",
"url":18,
"doc":"Find the bboxes of the binary image. The proposed algorithm is strictely equivalent to find the clusters of the binary image with a DBSCAN of distance 1, then to find the bboxes of each cluster. Parameters      binary : np.ndarray The c contiguous 2d image of boolean. max_size : int The max bounding boxe size, reject the strictly bigger bboxes. use_cv2 : bool, default=False If set to True, use the algorithm provided by cv2 rather than laueimproc's own algorithme. Be carefull, behavior is not the same! There are a few misses and the clusters next to each other on a diagonal are merged. The cv2 algorithm is about 3 times slowler than the compiled C version for high density, but still around 100 times faster than the pure python version. The cv2 algorithm doesn't release the GIL, making it difficult to multithread. Returns    - bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) They are sorted by ascending colums then lignes. This order optimizes the image cache acces. Examples     >>> import numpy as np >>> from laueimproc.improc.find_bboxes import find_bboxes >>> binary = np.array( 1, 0, 1, 0, 1],  . [0, 1, 0, 1, 0],  . [1, 0, 1, 0, 1 ,  . dtype=np.uint8) >>> find_bboxes(binary) tensor( 0, 0, 1, 1], [0, 2, 1, 1], [0, 4, 1, 1], [1, 1, 1, 1], [1, 3, 1, 1], [2, 0, 1, 1], [2, 2, 1, 1], [2, 4, 1, 1 , dtype=torch.int16) >>> binary = np.array( 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  . [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],  . [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],  . [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  . [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],  . [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],  . [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  . [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],  . [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],  . [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],  . [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  . [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  . [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  . [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],  . [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],  . [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  . [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],  . [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  . [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  . [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ,  . dtype=np.uint8) >>> find_bboxes(binary) tensor( 0, 0, 1, 1], [ 0, 6, 3, 1], [ 0, 8, 3, 2], [ 0, 11, 7, 10], [ 1, 1, 1, 1], [ 1, 3, 1, 2], [ 3, 1, 2, 1], [ 3, 3, 2, 2], [ 4, 4, 5, 5], [ 6, 0, 1, 3], [ 8, 0, 2, 3], [ 8, 8, 3, 3], [ 8, 12, 3, 4], [11, 0, 10, 7], [12, 8, 4, 3], [12, 14, 2, 2], [14, 12, 2, 2], [14, 16, 2, 2], [16, 14, 2, 2], [17, 17, 4, 4 , dtype=torch.int16) >>> def show():  . import matplotlib.pyplot as plt  . plt.imshow(  . binary,  . extent=(0, binary.shape[1], binary.shape[0], 0),  . cmap=\"gray\",  . interpolation=None,  . )  . bboxes = find_bboxes(binary)  . plt.plot(  . np.vstack  . bboxes[:, 1],  . bboxes[:, 1],  . bboxes[:, 1]+bboxes[:, 3],  . bboxes[:, 1]+bboxes[:, 3],  . bboxes[:, 1],  .  ,  . np.vstack  . bboxes[:, 0],  . bboxes[:, 0]+bboxes[:, 2],  . bboxes[:, 0]+bboxes[:, 2],  . bboxes[:, 0],  . bboxes[:, 0],  .  ,  . )  . plt.show()  . >>> show()  doctest: +SKIP >>> binary = (np.random.random 2048, 2048 > 0.75).view(np.uint8) >>> np.array_equal(find_bboxes(binary), find_bboxes(binary, _no_c=True True >>>",
"func":1
},
{
"ref":"laueimproc.improc.peaks_search",
"url":19,
"doc":"Atomic function for finding spots."
},
{
"ref":"laueimproc.improc.peaks_search.estimate_background",
"url":19,
"doc":"Estimate and Return the background of the image. Parameters      brut_image : torch.Tensor The 2d array brut image of the Laue diagram in float between 0 and 1. radius_font : float, default = 9 The structurant element radius used for the morphological opening. If it is not provided a circle of diameter 19 pixels is taken. More the radius or the kernel is big, better the estimation is, but slower the proccess is. Returns    - background : np.ndarray An estimation of the background.",
"func":1
},
{
"ref":"laueimproc.improc.peaks_search.peaks_search",
"url":19,
"doc":"Search all the spots roi in this diagram, return the roi tensor. Parameters      brut_image : torch.Tensor The 2d grayscale float32 image to annalyse. density : float, default=0.5 Correspond to the density of spots found. This value is normalized so that it can evolve 'linearly' between ]0, 1]. The smaller the value, the fewer spots will be captured. threshold : float, optional If provided, it replaces  density . It corresponds to the lowest density such as the pixel is considered as a spot. The threshold is appled after having removed the background. It evolves between ]0, 1[. radius_aglo : float, default = 2 The structurant element radius used for the aglomeration of close grain by morhpological dilatation applied on the thresholed image. If it is not provided a circle of diameter 5 pixels is taken. Bigger is the kernel, higher are the number of aglomerated spots but slower the proccess is.  kwargs : dict Transmitted to  estimate_background . Returns    - rois_no_background : bytearray The flatten float32 tensor data of each regions of interest without background. After unfolding and padding, the shape is (n, h, w). bboxes : torch.Tensor The positions of the corners point (0, 0) and the shape (i, j, h, w) of the roi of each spot in the brut_image, and the height and width of each roi. The shape is (n, 4) and the type is int. Examples     >>> from laueimproc.improc.peaks_search import peaks_search >>> from laueimproc.io import get_sample >>> from laueimproc.io.read import read_image >>> def print_stats(rois, bboxes):  . area = float bboxes[:, 2] bboxes[:, 3]).to(float).mean(  . print(f\"{len(rois)} bytes, {len(bboxes)} bboxes with average area of {area:.1f} pxl 2\")  . >>> img, _ = read_image(get_sample( >>> print_stats( peaks_search(img 89576 bytes, 240 bboxes with average area of 93.3 pxl 2 >>> >>> print_stats( peaks_search(img, density=0.3 9184 bytes, 64 bboxes with average area of 35.9 pxl 2 >>> print_stats( peaks_search(img, density=0.7 3179076 bytes, 15199 bboxes with average area of 52.3 pxl 2 >>> print_stats( peaks_search(img, threshold=0.01 23440 bytes, 125 bboxes with average area of 46.9 pxl 2 >>> print_stats( peaks_search(img, threshold=0.02 16540 bytes, 107 bboxes with average area of 38.6 pxl 2 >>> >>> print_stats( peaks_search(img, radius_aglo=1 58908 bytes, 241 bboxes with average area of 61.1 pxl 2 >>> print_stats( peaks_search(img, radius_aglo=5 229052 bytes, 235 bboxes with average area of 243.7 pxl 2 >>>",
"func":1
},
{
"ref":"laueimproc.classes",
"url":20,
"doc":"Defines the data structure used throughout the rest of the module."
},
{
"ref":"laueimproc.classes.Diagram",
"url":20,
"doc":"A Laue diagram image. Create a new diagram with appropriated metadata. Parameters      data : pathlike or arraylike The filename or the array/tensor use as a diagram. For memory management, it is better to provide a pathlike rather than an array."
},
{
"ref":"laueimproc.classes.Diagram.compute_rois_centroid",
"url":20,
"doc":"Compute the barycenter of each spots. Returns    - positions : torch.Tensor The 2 barycenter position for each roi. Each line corresponds to a spot and each column to an axis (shape (n, 2 . See  laueimproc.improc.spot.basic.compute_rois_centroid for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_centroid(  doctset: +ELLIPSIS tensor( 4.2637e+00, 5.0494e+00], [7.2805e-01, 2.6091e+01], [5.1465e-01, 1.9546e+03],  ., [1.9119e+03, 1.8759e+02], [1.9376e+03, 1.9745e+03], [1.9756e+03, 1.1794e+03 ) >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.compute_rois_inertia",
"url":20,
"doc":"Compute the inertia of the spot along the center of mass. Returns    - inertia : torch.Tensor The order 2 momentum centered along the barycenter (shape (n, . See  laueimproc.improc.spot.basic.compute_rois_inertia for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_inertia(  doctset: +ELLIPSIS tensor([4.0506e+00, 9.7974e-02, 1.6362e-01, 2.1423e-03, 4.8515e-02, 2.6657e-01, 9.9830e-02, 1.7221e-03, 2.8771e-02, 5.3754e-03, 2.4759e-02, 8.7665e-01, 5.1118e-02, 9.6223e-02, 3.0648e-01, 1.1362e-01, 2.6335e-01, 1.1388e-01,  ., 4.8586e+00, 4.4373e+00, 1.3409e-01, 3.1009e-01, 1.5538e-02, 8.4468e-03, 1.5235e+00, 1.0291e+01, 8.1125e+00, 5.4182e-01, 1.4235e+00, 4.4005e+00, 7.0188e+00, 2.5770e+00, 1.1837e+01, 7.0610e+00, 3.6725e+00, 2.2335e+01]) >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.compute_rois_max",
"url":20,
"doc":"Get the intensity and the position of the hottest pixel for each roi. Returns    - pos1_pos2_imax : torch.Tensor The concatenation of the colum vectors of the argmax and the intensity (shape (n, 3 . See  laueimproc.improc.spot.basic.compute_rois_max for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_max(  doctset: +ELLIPSIS tensor( 2.5000e+00, 5.5000e+00, 9.2165e-03], [5.0000e-01, 2.2500e+01, 1.3581e-03], [5.0000e-01, 1.9545e+03, 1.3123e-02],  ., [1.9125e+03, 1.8750e+02, 1.1250e-01], [1.9375e+03, 1.9745e+03, 6.0273e-02], [1.9755e+03, 1.1795e+03, 2.9212e-01 ) >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.compute_rois_nb_peaks",
"url":20,
"doc":"Find the number of extremums in each roi. Returns    - nbr_of_peaks : torch.Tensor The number of extremums (shape (n, . See  laueimproc.improc.spot.extrema.compute_rois_nb_peaks for more details. Notes   - No noise filtering. Doesn't detect shoulders. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_nb_peaks( tensor([23, 4, 3, 2, 6, 3, 1, 3, 1, 4, 1, 2, 3, 1, 1, 3, 3, 3, 2, 7, 4, 1, 1, 3, 3, 2, 4, 2, 1, 2, 1, 2, 2, 2, 2, 2, 6, 2, 1, 2, 4, 2, 3, 4, 1, 1, 4, 3, 1, 4, 4, 5, 3, 3, 1, 5, 3, 2, 6, 1, 2, 3, 1, 4, 6, 3, 6, 2, 2, 5, 7, 5, 3, 2, 3, 1, 2, 6, 8, 2, 3, 3, 3, 3, 2, 4, 2, 1, 1, 4, 4, 1, 4, 2, 5, 4, 3, 3, 1, 2, 1, 3, 4, 2, 5, 5, 7, 2, 6, 3, 5, 2, 2, 2, 7, 2, 3, 5, 3, 1, 4, 8, 3, 3, 4, 3, 4, 2, 5, 1, 7, 3, 5, 4, 10, 2, 3, 3, 4, 3, 5, 8, 2, 5, 8, 3, 2, 3, 5, 5, 5, 3, 4, 6, 2, 4, 1, 3, 4, 2, 3, 4, 5, 5, 2, 4, 6, 4, 4, 1, 2, 4, 4, 3, 13, 13, 3, 6, 3, 2, 1, 4, 6, 3, 2, 2, 16, 3, 1, 2, 2, 4, 1, 3, 7, 4, 1, 5, 4, 7, 5, 1, 6, 6, 2, 4, 3, 9, 3, 2, 3, 2, 8, 5, 3, 3, 3, 3, 10, 9, 2, 2, 5, 4, 3, 3, 2, 1, 3, 6, 11, 2, 4, 4, 6, 2, 7, 5, 2, 10], dtype=torch.int16) >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.compute_rois_pca",
"url":20,
"doc":"Compute the pca on each spot. Returns    - std1_std2_theta : torch.Tensor The concatenation of the colum vectors of the two std and the angle (shape (n, 3 . See  laueimproc.improc.spot.pca.compute_rois_pca for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_pca(  doctset: +ELLIPSIS tensor( 3.5272e+00, 2.4630e+00, -8.8861e-01], [ 2.8604e+00, 4.8286e-01, -1.5335e+00], [ 1.6986e+00, 1.5142e-01, -1.5628e+00],  ., [ 2.1297e+00, 1.7486e+00, -2.6128e-01], [ 2.0484e+00, 1.7913e+00, 4.9089e-01], [ 2.6780e+00, 1.9457e+00, 9.9400e-02 ) >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.compute_rois_sum",
"url":20,
"doc":"Sum the intensities of the pixels for each roi. Returns    - total_intensity : torch.Tensor The intensity of each roi, sum of the pixels (shape (n, . See  laueimproc.improc.spot.basic.compute_rois_sum for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_sum(  doctset: +ELLIPSIS tensor([2.1886e-01, 1.1643e-02, 5.6260e-02, 1.6938e-03, 9.8726e-03, 6.1647e-02, 3.5279e-02, 3.1891e-03, 1.0071e-02, 2.2889e-03, 1.0285e-02, 1.9760e-01, 1.7258e-02, 3.3402e-02, 9.1829e-02, 3.1510e-02, 8.4550e-02, 4.0864e-02,  ., 9.2822e-01, 9.7104e-01, 4.1733e-02, 9.4377e-02, 5.2491e-03, 4.2115e-03, 3.9217e-01, 1.5907e+00, 1.1802e+00, 1.4968e-01, 4.0696e-01, 6.3442e-01, 1.3559e+00, 6.0548e-01, 1.7116e+00, 9.2990e-01, 4.9596e-01, 2.0383e+00]) >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.fit_gaussians_em",
"url":20,
"doc":"Fit each roi by \\(K\\) gaussians using the EM algorithm. See  laueimproc.gmm for terminology, and  laueimproc.gmm.fit.fit_em for the algo description. Parameters      nbr_clusters, nbr_tries, aic, bic, log_likelihood Transmitted to  laueimproc.improc.spot.fit.fit_gaussians_em . Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). eta : torch.Tensor The relative mass \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> mean, cov, eta, _ = diagram.fit_gaussians_em(nbr_clusters=3, nbr_tries=4) >>> mean.shape, cov.shape, eta.shape (torch.Size([240, 3, 2]), torch.Size([240, 3, 2, 2]), torch.Size([240, 3] >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.fit_gaussians_mse",
"url":20,
"doc":"Fit each roi by \\(K\\) gaussians by minimising the mean square error. See  laueimproc.gmm for terminology, and  laueimproc.gmm.fit.fit_mse for the algo description. Parameters      nbr_clusters, nbr_tries, mse Transmitted to  laueimproc.improc.spot.fit.fit_gaussians_mse . Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). mag : torch.Tensor The absolute magnitude \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> mean, cov, mag, _ = diagram.fit_gaussians_mse(nbr_clusters=3, nbr_tries=4) >>> mean.shape, cov.shape, mag.shape (torch.Size([240, 3, 2]), torch.Size([240, 3, 2, 2]), torch.Size([240, 3] >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.add_property",
"url":1,
"doc":"Add a property to the diagram. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the diagram state changed.",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.areas",
"url":1,
"doc":"Return the int32 area of each bboxes. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.areas) None >>> diagram.find_spots() >>> print(diagram.areas)  doctest: +ELLIPSIS tensor([289, 36, 33, 20, 40, 81, 49, 15, 36, 25, 36, 110, 49, 56, 64, 56, 72, 56, 90, 143, 64, 42, 36, 25, 169, 110, 81, 64, 100, 49, 36, 42, 121, 36, 36, 121, 81, 56, 72, 80, 110, 56,  ., 100, 25, 225, 182, 72, 156, 90, 25, 320, 288, 144, 143, 240, 208, 64, 81, 25, 25, 144, 323, 300, 90, 144, 240, 270, 168, 352, 270, 210, 456], dtype=torch.int32) >>>"
},
{
"ref":"laueimproc.classes.Diagram.bboxes",
"url":1,
"doc":"Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width). Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.bboxes) None >>> diagram.find_spots() >>> print(diagram.bboxes)  doctest: +ELLIPSIS tensor( 0, 0, 17, 17], [ 0, 20, 3, 12], [ 0, 1949, 3, 11],  ., [1903, 180, 18, 15], [1930, 1967, 15, 14], [1963, 1170, 24, 19 , dtype=torch.int16) >>>"
},
{
"ref":"laueimproc.classes.Diagram.clone",
"url":1,
"doc":"Instanciate a new identical diagram. Parameters      deep : boolean, default=True If True, the memory of the new created diagram object is totally independ of this object (slow but safe). Otherwise,  self.image ,  self.rois ,  properties and  cached data share the same memory view (Tensor). So modifying one of these attributes in one diagram will modify the same attribute in the other diagram. It if realy faster but not safe. cache : boolean, default=True Copy the cache into the new diagram if True (default), or live the cache empty if False. Returns    - Diagram The new copy of self. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram_bis = diagram.clone() >>> assert id(diagram) != id(diagram_bis) >>> assert diagram.state  diagram_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.compress",
"url":1,
"doc":"Delete attributes and elements in the cache. Paremeters      size : int The quantity of bytes to remove from the cache. Returns    - removed : int The number of bytes removed from the cache.",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.file",
"url":1,
"doc":"Return the absolute file path to the image, if it is provided. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> BaseDiagram(get_sample( .file  doctest: +ELLIPSIS PosixPath('/ ./laueimproc/io/ge.jp2') >>>"
},
{
"ref":"laueimproc.classes.Diagram.filter_spots",
"url":1,
"doc":"Keep only the given spots, delete the rest. This method can be used for filtering or sorting spots. Parameters      criteria : arraylike The list of the indices of the spots to keep (negatives indices are allow), or the boolean vector with True for keeping the spot, False otherwise like a mask. msg : str The message to happend to the history. inplace : boolean, default=True If True, modify the diagram self (no copy) and return a reference to self. If False, first clone the diagram, then apply the selection on the new diagram, It create a checkpoint (real copy) (but it is slowler). Returns    - filtered_diagram : BaseDiagram Return None if inplace is True, or a filtered clone of self otherwise. Examples     >>> from pprint import pprint >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> indices = torch.arange(0, len(diagram), 2) >>> diagram.filter_spots(indices, \"keep even spots\") >>> cond = diagram.bboxes[:, 1]  >> diag_final = diagram.filter_spots(cond, \"keep spots on left\", inplace=False) >>> pprint(diagram.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots'] >>> pprint(diag_final.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots', '120 to 63 spots: keep spots on left'] >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.find_spots",
"url":1,
"doc":"Search all the spots in this diagram, store the result into the  spots attribute. Parameters      kwargs : dict Transmitted to  laueimproc.improc.peaks_search.peaks_search .",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.get_properties",
"url":1,
"doc":"Return all the available properties.",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.get_property",
"url":1,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> diagram.add_property(\"prop2\", value=\"any python object 2\") >>> diagram.get_property(\"prop1\") 'any python object 1' >>> diagram.get_property(\"prop2\") 'any python object 2' >>> diagram.find_spots()  change state >>> diagram.get_property(\"prop1\") 'any python object 1' >>> try:  . diagram.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the diagram has changed\" >>> try:  . diagram.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.history",
"url":1,
"doc":"Return the actions performed on the Diagram since the initialisation."
},
{
"ref":"laueimproc.classes.Diagram.is_init",
"url":1,
"doc":"Return True if the diagram has been initialized. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.is_init() False >>> diagram.find_spots() >>> diagram.is_init() True >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.image",
"url":1,
"doc":"Return the complete image of the diagram. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.image.shape torch.Size([2018, 2016]) >>> diagram.image.min() >= 0 tensor(True) >>> diagram.image.max()  >>"
},
{
"ref":"laueimproc.classes.Diagram.plot",
"url":1,
"doc":"Prepare for display the diagram and the spots. Parameters      disp : matplotlib.figure.Figure or matplotlib.axes.Axes The matplotlib figure to complete. vmin : float, optional The minimum intensity ploted. vmax : float, optional The maximum intensity ploted. show_axis: boolean, default=True Display the label and the axis if True. show_bboxes: boolean, default=True Draw all the bounding boxes if True. show_image: boolean, default=True Display the image if True, dont call imshow otherwise. Returns    - matplotlib.axes.Axes Filled graphical widget. Notes   - It doesn't create the figure and call show. Use  self.show() to Display the diagram from scratch.",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.rawrois",
"url":1,
"doc":"Return the tensor of the raw rois of the spots."
},
{
"ref":"laueimproc.classes.Diagram.rois",
"url":1,
"doc":"Return the tensor of the provided rois of the spots. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> diagram.rawrois.shape torch.Size([240, 24, 19]) >>> diagram.rois.shape torch.Size([240, 24, 19]) >>> diagram.rois.mean()  >>"
},
{
"ref":"laueimproc.classes.Diagram.set_spots",
"url":1,
"doc":"Set the new spots as the current spots, reset the history and the cache. Paremeters      new_spots : tuple Can be an over diagram of type Diagram. Can be an arraylike of bounding boxes n  (anchor_i, anchor_j, height, width). Can be an anchor and a picture n  (anchor_i, anchor_j, roi). It can also be a combination of the above elements to unite the spots. Examples     >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> >>>  from diagram >>> diagram_ref = BaseDiagram(get_sample( >>> diagram_ref.find_spots() >>> diagram_ref.filter_spots(range(10 >>> >>>  from bboxes >>> bboxes =  300, 500, 10, 15], [600, 700, 20, 15 >>> >>>  from anchor and roi >>> spots =  400, 600, torch.zeros 10, 15 ], [700, 800, torch.zeros 20, 15  >>> >>>  union of the spots >>> diagram = BaseDiagram(get_sample( >>> diagram.set_spots(diagram_ref, bboxes, spots) >>> len(diagram) 14 >>>",
"func":1
},
{
"ref":"laueimproc.classes.Diagram.state",
"url":1,
"doc":"Return a hash of the diagram. If two diagrams gots the same state, it means they are the same. The hash take in consideration the internal state of the diagram. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.state '9570d3b8743757aa3bf89a42bc4911de' >>> diagram.find_spots() >>> diagram.state '539428b799f2640150aab633de49b695' >>>"
},
{
"ref":"laueimproc.classes.DiagramsDataset",
"url":20,
"doc":"Link Diagrams together. Initialise the dataset. Parameters       diagram_refs : tuple The diagram references, transmitted to  add_diagrams . diag2ind : callable, default=laueimproc.classes.base_dataset.default_diag2ind The function to associate an index to a diagram. If provided, this function has to be pickleable. It has to take only one positional argument (a simple Diagram instance) and to return a positive integer. diag2scalars : callable, optional If provided, it allows you to select diagram from physical parameters. \\(f:I\\to\\mathbb{R}^n\\), an injective application. \\(I\\subset\\mathbb{N}\\) is the set of the indices of the diagrams in the dataset. \\(n\\) corresponds to the number of physical parameters. For example the beam positon on the sample or, more generally, a vector of scalar parameters like time, voltage, temperature, pressure . The function must match  laueimproc.ml.dataset_dist.check_diag2scalars_typing ."
},
{
"ref":"laueimproc.classes.DiagramsDataset.compute_mean_image",
"url":20,
"doc":"Compute the average image. Based on  laueimproc.immix.mean.mean_stack . Returns    - torch.Tensor The average image of all the images contained in this dataset.",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.compute_inter_image",
"url":20,
"doc":"Compute the median, first quartile, third quartile or everything in between. Parameters       args,  kwargs Transmitted to  laueimproc.immix.inter.sort_stack or  laueimproc.immix.inter.snowflake_stack . method : str The algorithm used. It can be  sort for the naive accurate algorithm, or  snowflake for a faster algorithm on big dataset, but less accurate. Returns    - torch.Tensor The 2d float32 grayscale image.",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.select_closest",
"url":20,
"doc":"Select the closest diagram to a given set of phisical parameters. Parameters      point, tol, scale Transmitted to  laueimproc.ml.dataset_dist.select_closest . no_raise : boolean, default = False If set to True, return None rather throwing a LookupError exception. Returns    - closet_diag :  laueimproc.classes.diagram.Diagram The closet diagram. Examples     >>> from laueimproc.classes.dataset import DiagramsDataset >>> from laueimproc.io import get_samples >>> def position(index: int) -> tuple[int, int]:  . return divmod(index, 10)  . >>> dataset = DiagramsDataset(get_samples(), diag2scalars=position) >>> dataset.select_closest 4.1, 4.9), tol=(0.2, 0.2 Diagram(img_45.jp2) >>>",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.select_closests",
"url":20,
"doc":"Select the diagrams matching a given interval of phisical parameters. Parameters      point, tol, scale Transmitted to  laueimproc.ml.dataset_dist.select_closests . Returns    - sub_dataset :  laueimproc.classes.dataset.DiagramsDataset The frozen view of this dataset containing only the diagram matching the constraints. Examples     >>> from laueimproc.classes.dataset import DiagramsDataset >>> from laueimproc.io import get_samples >>> def position(index: int) -> tuple[int, int]:  . return divmod(index, 10)  . >>> dataset = DiagramsDataset(get_samples(), diag2scalars=position) >>> dataset.select_closests 4.1, 4.9), tol=(1.2, 1.2  >>>",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.train_spot_classifier",
"url":20,
"doc":"Train a variational autoencoder classifier with the spots in the diagrams. It is a non supervised neuronal network. Parameters      model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier, optional If provided, this model will be trained again and returned. shape : float or tuple[int, int], default=0.95 The model input spot shape in numpy convention (height, width). If a percentage is given, the shape is founded with  laueimproc.nn.loader.find_shape . space : float, default = 3.0 The non penalized spreading area half size. Transmitted to  laueimproc.nn.vae_spot_classifier.VAESpotClassifier . intensity_sensitive, scale_sensitive : boolean, default=True Transmitted to  laueimproc.nn.vae_spot_classifier.VAESpotClassifier . epoch, lr, fig Transmitted to  laueimproc.nn.train.train_vae_spot_classifier . Returns    - classifier: laueimproc.nn.vae_spot_classifier.VAESpotClassifier The trained model. Examples     >>> import laueimproc >>> def init(diagram: laueimproc.Diagram):  . diagram.find_spots()  . diagram.filter_spots(range(10  to reduce the number of spots  . >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples( >>> _ = dataset.apply(init) >>> model = dataset.train_spot_classifier(epoch=2) >>>",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.add_property",
"url":2,
"doc":"Add a property to the dataset. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the dataset state changed.",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.add_diagram",
"url":2,
"doc":"Append a new diagram into the dataset. Parameters      new_diagram : laueimproc.classes.diagram.Diagram The new instanciated diagram not already present in the dataset. Raises    LookupError If the diagram is already present in the dataset.",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.add_diagrams",
"url":2,
"doc":"Append the new diagrams into the datset. Parameters      new_diagrams The diagram references, they can be of this natures:  laueimproc.classes.diagram.Diagram : Could be a simple Diagram instance.  iterable : An iterable of any of the types specified above. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> file = min(get_samples().iterdir( >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(Diagram(file  from Diagram instance >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(file)  from filename (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(file  from filename (str) >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(get_samples(  from folder (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(get_samples( )  from folder (str) >>> dataset  >>> >>>  from iterable (DiagramDataset, list, tuple,  .) >>>  ok with nested iterables >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams([Diagram(f) for f in get_samples().iterdir()]) >>> dataset  >>>",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.apply",
"url":2,
"doc":"Apply an operation in all the diagrams of the dataset. Parameters      func : callable A function that take a diagram and optionaly other parameters, and return anything. The function can modify the diagram inplace. It has to be pickaleable. args : tuple, optional Positional arguments transmitted to the provided func. kwargs : dict, optional Keyword arguments transmitted to the provided func. Returns    - res : dict The result of the function for each diagrams. To each diagram index, associate the result. Notes   - This function will be automaticaly applided on the new diagrams, but the result is throw. Examples     >>> from pprint import pprint >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( [ 10]  subset to go faster >>> def peak_search(diagram: Diagram, density: float) -> int:  .  'Return the number of spots. '  . diagram.find_spots(density=density)  . return len(diagram)  . >>> res = dataset.apply(peak_search, args=(0.5, >>> pprint(res) {0: 204, 10: 547, 20: 551, 30: 477, 40: 404, 50: 537, 60: 2121, 70: 274, 80: 271, 90: 481} >>>",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.autosave",
"url":2,
"doc":"Manage the dataset recovery. This allows the dataset to be backuped at regular time intervals. Parameters      filename : pathlike The name of the persistant file, transmitted to  laueimproc.io.save_dataset.write_dataset . delay : float or str, optional If provided and > 0, automatic checkpoint will be performed. it corresponds to the time interval between two check points. The supported formats are defined in  laueimproc.common.time2sec .",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.clone",
"url":2,
"doc":"Instanciate a new identical dataset. Parameters       kwargs : dict Transmitted to  laueimproc.classes.base_diagram.BaseDiagram.clone . Returns    - BaseDiagramsDataset The new copy of self. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset_bis = dataset.clone() >>> assert id(dataset) != id(dataset_bis) >>> assert dataset.state  dataset_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.flush",
"url":2,
"doc":"Extract finished thread diagrams.",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.get_property",
"url":2,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> dataset.add_property(\"prop2\", value=\"any python object 2\") >>> dataset.get_property(\"prop1\") 'any python object 1' >>> dataset.get_property(\"prop2\") 'any python object 2' >>> dataset = dataset[:1]  change state >>> dataset.get_property(\"prop1\") 'any python object 1' >>> try:  . dataset.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the dataset has changed\" >>> try:  . dataset.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.get_scalars",
"url":2,
"doc":"Return the scalars values of each diagrams given by  diag2scalars .",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.indices",
"url":2,
"doc":"Return the sorted map diagram indices currently reachable in the dataset. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset[-10:10:-5].indices [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90] >>>"
},
{
"ref":"laueimproc.classes.DiagramsDataset.state",
"url":2,
"doc":"Return a hash of the dataset. If two datasets gots the same state, it means they are the same. The hash take in consideration the indices of the diagrams and the functions applyed. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:].state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:1].state '1de1605a297bafd22a886de7058cae81' >>>"
},
{
"ref":"laueimproc.classes.DiagramsDataset.restore",
"url":2,
"doc":"Restore the dataset content from the backup file. Do nothing if the file doesn't exists. Based on  laueimproc.io.save_dataset.restore_dataset . Parameters      filename : pathlike The name of the persistant file.",
"func":1
},
{
"ref":"laueimproc.classes.DiagramsDataset.run",
"url":2,
"doc":"Run asynchronousely in a child thread, called by self.start().",
"func":1
},
{
"ref":"laueimproc.classes.diagram",
"url":21,
"doc":"Define the data sructure of a single Laue diagram image."
},
{
"ref":"laueimproc.classes.diagram.Diagram",
"url":21,
"doc":"A Laue diagram image. Create a new diagram with appropriated metadata. Parameters      data : pathlike or arraylike The filename or the array/tensor use as a diagram. For memory management, it is better to provide a pathlike rather than an array."
},
{
"ref":"laueimproc.classes.diagram.Diagram.compute_rois_centroid",
"url":21,
"doc":"Compute the barycenter of each spots. Returns    - positions : torch.Tensor The 2 barycenter position for each roi. Each line corresponds to a spot and each column to an axis (shape (n, 2 . See  laueimproc.improc.spot.basic.compute_rois_centroid for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_centroid(  doctset: +ELLIPSIS tensor( 4.2637e+00, 5.0494e+00], [7.2805e-01, 2.6091e+01], [5.1465e-01, 1.9546e+03],  ., [1.9119e+03, 1.8759e+02], [1.9376e+03, 1.9745e+03], [1.9756e+03, 1.1794e+03 ) >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.compute_rois_inertia",
"url":21,
"doc":"Compute the inertia of the spot along the center of mass. Returns    - inertia : torch.Tensor The order 2 momentum centered along the barycenter (shape (n, . See  laueimproc.improc.spot.basic.compute_rois_inertia for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_inertia(  doctset: +ELLIPSIS tensor([4.0506e+00, 9.7974e-02, 1.6362e-01, 2.1423e-03, 4.8515e-02, 2.6657e-01, 9.9830e-02, 1.7221e-03, 2.8771e-02, 5.3754e-03, 2.4759e-02, 8.7665e-01, 5.1118e-02, 9.6223e-02, 3.0648e-01, 1.1362e-01, 2.6335e-01, 1.1388e-01,  ., 4.8586e+00, 4.4373e+00, 1.3409e-01, 3.1009e-01, 1.5538e-02, 8.4468e-03, 1.5235e+00, 1.0291e+01, 8.1125e+00, 5.4182e-01, 1.4235e+00, 4.4005e+00, 7.0188e+00, 2.5770e+00, 1.1837e+01, 7.0610e+00, 3.6725e+00, 2.2335e+01]) >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.compute_rois_max",
"url":21,
"doc":"Get the intensity and the position of the hottest pixel for each roi. Returns    - pos1_pos2_imax : torch.Tensor The concatenation of the colum vectors of the argmax and the intensity (shape (n, 3 . See  laueimproc.improc.spot.basic.compute_rois_max for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_max(  doctset: +ELLIPSIS tensor( 2.5000e+00, 5.5000e+00, 9.2165e-03], [5.0000e-01, 2.2500e+01, 1.3581e-03], [5.0000e-01, 1.9545e+03, 1.3123e-02],  ., [1.9125e+03, 1.8750e+02, 1.1250e-01], [1.9375e+03, 1.9745e+03, 6.0273e-02], [1.9755e+03, 1.1795e+03, 2.9212e-01 ) >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.compute_rois_nb_peaks",
"url":21,
"doc":"Find the number of extremums in each roi. Returns    - nbr_of_peaks : torch.Tensor The number of extremums (shape (n, . See  laueimproc.improc.spot.extrema.compute_rois_nb_peaks for more details. Notes   - No noise filtering. Doesn't detect shoulders. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_nb_peaks( tensor([23, 4, 3, 2, 6, 3, 1, 3, 1, 4, 1, 2, 3, 1, 1, 3, 3, 3, 2, 7, 4, 1, 1, 3, 3, 2, 4, 2, 1, 2, 1, 2, 2, 2, 2, 2, 6, 2, 1, 2, 4, 2, 3, 4, 1, 1, 4, 3, 1, 4, 4, 5, 3, 3, 1, 5, 3, 2, 6, 1, 2, 3, 1, 4, 6, 3, 6, 2, 2, 5, 7, 5, 3, 2, 3, 1, 2, 6, 8, 2, 3, 3, 3, 3, 2, 4, 2, 1, 1, 4, 4, 1, 4, 2, 5, 4, 3, 3, 1, 2, 1, 3, 4, 2, 5, 5, 7, 2, 6, 3, 5, 2, 2, 2, 7, 2, 3, 5, 3, 1, 4, 8, 3, 3, 4, 3, 4, 2, 5, 1, 7, 3, 5, 4, 10, 2, 3, 3, 4, 3, 5, 8, 2, 5, 8, 3, 2, 3, 5, 5, 5, 3, 4, 6, 2, 4, 1, 3, 4, 2, 3, 4, 5, 5, 2, 4, 6, 4, 4, 1, 2, 4, 4, 3, 13, 13, 3, 6, 3, 2, 1, 4, 6, 3, 2, 2, 16, 3, 1, 2, 2, 4, 1, 3, 7, 4, 1, 5, 4, 7, 5, 1, 6, 6, 2, 4, 3, 9, 3, 2, 3, 2, 8, 5, 3, 3, 3, 3, 10, 9, 2, 2, 5, 4, 3, 3, 2, 1, 3, 6, 11, 2, 4, 4, 6, 2, 7, 5, 2, 10], dtype=torch.int16) >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.compute_rois_pca",
"url":21,
"doc":"Compute the pca on each spot. Returns    - std1_std2_theta : torch.Tensor The concatenation of the colum vectors of the two std and the angle (shape (n, 3 . See  laueimproc.improc.spot.pca.compute_rois_pca for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_pca(  doctset: +ELLIPSIS tensor( 3.5272e+00, 2.4630e+00, -8.8861e-01], [ 2.8604e+00, 4.8286e-01, -1.5335e+00], [ 1.6986e+00, 1.5142e-01, -1.5628e+00],  ., [ 2.1297e+00, 1.7486e+00, -2.6128e-01], [ 2.0484e+00, 1.7913e+00, 4.9089e-01], [ 2.6780e+00, 1.9457e+00, 9.9400e-02 ) >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.compute_rois_sum",
"url":21,
"doc":"Sum the intensities of the pixels for each roi. Returns    - total_intensity : torch.Tensor The intensity of each roi, sum of the pixels (shape (n, . See  laueimproc.improc.spot.basic.compute_rois_sum for more details. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> print(diagram.compute_rois_sum(  doctset: +ELLIPSIS tensor([2.1886e-01, 1.1643e-02, 5.6260e-02, 1.6938e-03, 9.8726e-03, 6.1647e-02, 3.5279e-02, 3.1891e-03, 1.0071e-02, 2.2889e-03, 1.0285e-02, 1.9760e-01, 1.7258e-02, 3.3402e-02, 9.1829e-02, 3.1510e-02, 8.4550e-02, 4.0864e-02,  ., 9.2822e-01, 9.7104e-01, 4.1733e-02, 9.4377e-02, 5.2491e-03, 4.2115e-03, 3.9217e-01, 1.5907e+00, 1.1802e+00, 1.4968e-01, 4.0696e-01, 6.3442e-01, 1.3559e+00, 6.0548e-01, 1.7116e+00, 9.2990e-01, 4.9596e-01, 2.0383e+00]) >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.fit_gaussians_em",
"url":21,
"doc":"Fit each roi by \\(K\\) gaussians using the EM algorithm. See  laueimproc.gmm for terminology, and  laueimproc.gmm.fit.fit_em for the algo description. Parameters      nbr_clusters, nbr_tries, aic, bic, log_likelihood Transmitted to  laueimproc.improc.spot.fit.fit_gaussians_em . Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). eta : torch.Tensor The relative mass \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> mean, cov, eta, _ = diagram.fit_gaussians_em(nbr_clusters=3, nbr_tries=4) >>> mean.shape, cov.shape, eta.shape (torch.Size([240, 3, 2]), torch.Size([240, 3, 2, 2]), torch.Size([240, 3] >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.fit_gaussians_mse",
"url":21,
"doc":"Fit each roi by \\(K\\) gaussians by minimising the mean square error. See  laueimproc.gmm for terminology, and  laueimproc.gmm.fit.fit_mse for the algo description. Parameters      nbr_clusters, nbr_tries, mse Transmitted to  laueimproc.improc.spot.fit.fit_gaussians_mse . Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape (n, \\(K\\), 2). In the absolute diagram base. cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape (n, \\(K\\), 2, 2). mag : torch.Tensor The absolute magnitude \\(\\eta\\). Shape (n, \\(K\\ . infodict : dict[str] A dictionary of optional outputs. Examples     >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_sample >>> diagram = Diagram(get_sample( >>> diagram.find_spots() >>> mean, cov, mag, _ = diagram.fit_gaussians_mse(nbr_clusters=3, nbr_tries=4) >>> mean.shape, cov.shape, mag.shape (torch.Size([240, 3, 2]), torch.Size([240, 3, 2, 2]), torch.Size([240, 3] >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.add_property",
"url":1,
"doc":"Add a property to the diagram. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the diagram state changed.",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.areas",
"url":1,
"doc":"Return the int32 area of each bboxes. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.areas) None >>> diagram.find_spots() >>> print(diagram.areas)  doctest: +ELLIPSIS tensor([289, 36, 33, 20, 40, 81, 49, 15, 36, 25, 36, 110, 49, 56, 64, 56, 72, 56, 90, 143, 64, 42, 36, 25, 169, 110, 81, 64, 100, 49, 36, 42, 121, 36, 36, 121, 81, 56, 72, 80, 110, 56,  ., 100, 25, 225, 182, 72, 156, 90, 25, 320, 288, 144, 143, 240, 208, 64, 81, 25, 25, 144, 323, 300, 90, 144, 240, 270, 168, 352, 270, 210, 456], dtype=torch.int32) >>>"
},
{
"ref":"laueimproc.classes.diagram.Diagram.bboxes",
"url":1,
"doc":"Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width). Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.bboxes) None >>> diagram.find_spots() >>> print(diagram.bboxes)  doctest: +ELLIPSIS tensor( 0, 0, 17, 17], [ 0, 20, 3, 12], [ 0, 1949, 3, 11],  ., [1903, 180, 18, 15], [1930, 1967, 15, 14], [1963, 1170, 24, 19 , dtype=torch.int16) >>>"
},
{
"ref":"laueimproc.classes.diagram.Diagram.clone",
"url":1,
"doc":"Instanciate a new identical diagram. Parameters      deep : boolean, default=True If True, the memory of the new created diagram object is totally independ of this object (slow but safe). Otherwise,  self.image ,  self.rois ,  properties and  cached data share the same memory view (Tensor). So modifying one of these attributes in one diagram will modify the same attribute in the other diagram. It if realy faster but not safe. cache : boolean, default=True Copy the cache into the new diagram if True (default), or live the cache empty if False. Returns    - Diagram The new copy of self. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram_bis = diagram.clone() >>> assert id(diagram) != id(diagram_bis) >>> assert diagram.state  diagram_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.compress",
"url":1,
"doc":"Delete attributes and elements in the cache. Paremeters      size : int The quantity of bytes to remove from the cache. Returns    - removed : int The number of bytes removed from the cache.",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.file",
"url":1,
"doc":"Return the absolute file path to the image, if it is provided. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> BaseDiagram(get_sample( .file  doctest: +ELLIPSIS PosixPath('/ ./laueimproc/io/ge.jp2') >>>"
},
{
"ref":"laueimproc.classes.diagram.Diagram.filter_spots",
"url":1,
"doc":"Keep only the given spots, delete the rest. This method can be used for filtering or sorting spots. Parameters      criteria : arraylike The list of the indices of the spots to keep (negatives indices are allow), or the boolean vector with True for keeping the spot, False otherwise like a mask. msg : str The message to happend to the history. inplace : boolean, default=True If True, modify the diagram self (no copy) and return a reference to self. If False, first clone the diagram, then apply the selection on the new diagram, It create a checkpoint (real copy) (but it is slowler). Returns    - filtered_diagram : BaseDiagram Return None if inplace is True, or a filtered clone of self otherwise. Examples     >>> from pprint import pprint >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> indices = torch.arange(0, len(diagram), 2) >>> diagram.filter_spots(indices, \"keep even spots\") >>> cond = diagram.bboxes[:, 1]  >> diag_final = diagram.filter_spots(cond, \"keep spots on left\", inplace=False) >>> pprint(diagram.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots'] >>> pprint(diag_final.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots', '120 to 63 spots: keep spots on left'] >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.find_spots",
"url":1,
"doc":"Search all the spots in this diagram, store the result into the  spots attribute. Parameters      kwargs : dict Transmitted to  laueimproc.improc.peaks_search.peaks_search .",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.get_properties",
"url":1,
"doc":"Return all the available properties.",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.get_property",
"url":1,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> diagram.add_property(\"prop2\", value=\"any python object 2\") >>> diagram.get_property(\"prop1\") 'any python object 1' >>> diagram.get_property(\"prop2\") 'any python object 2' >>> diagram.find_spots()  change state >>> diagram.get_property(\"prop1\") 'any python object 1' >>> try:  . diagram.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the diagram has changed\" >>> try:  . diagram.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.history",
"url":1,
"doc":"Return the actions performed on the Diagram since the initialisation."
},
{
"ref":"laueimproc.classes.diagram.Diagram.is_init",
"url":1,
"doc":"Return True if the diagram has been initialized. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.is_init() False >>> diagram.find_spots() >>> diagram.is_init() True >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.image",
"url":1,
"doc":"Return the complete image of the diagram. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.image.shape torch.Size([2018, 2016]) >>> diagram.image.min() >= 0 tensor(True) >>> diagram.image.max()  >>"
},
{
"ref":"laueimproc.classes.diagram.Diagram.plot",
"url":1,
"doc":"Prepare for display the diagram and the spots. Parameters      disp : matplotlib.figure.Figure or matplotlib.axes.Axes The matplotlib figure to complete. vmin : float, optional The minimum intensity ploted. vmax : float, optional The maximum intensity ploted. show_axis: boolean, default=True Display the label and the axis if True. show_bboxes: boolean, default=True Draw all the bounding boxes if True. show_image: boolean, default=True Display the image if True, dont call imshow otherwise. Returns    - matplotlib.axes.Axes Filled graphical widget. Notes   - It doesn't create the figure and call show. Use  self.show() to Display the diagram from scratch.",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.rawrois",
"url":1,
"doc":"Return the tensor of the raw rois of the spots."
},
{
"ref":"laueimproc.classes.diagram.Diagram.rois",
"url":1,
"doc":"Return the tensor of the provided rois of the spots. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> diagram.rawrois.shape torch.Size([240, 24, 19]) >>> diagram.rois.shape torch.Size([240, 24, 19]) >>> diagram.rois.mean()  >>"
},
{
"ref":"laueimproc.classes.diagram.Diagram.set_spots",
"url":1,
"doc":"Set the new spots as the current spots, reset the history and the cache. Paremeters      new_spots : tuple Can be an over diagram of type Diagram. Can be an arraylike of bounding boxes n  (anchor_i, anchor_j, height, width). Can be an anchor and a picture n  (anchor_i, anchor_j, roi). It can also be a combination of the above elements to unite the spots. Examples     >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> >>>  from diagram >>> diagram_ref = BaseDiagram(get_sample( >>> diagram_ref.find_spots() >>> diagram_ref.filter_spots(range(10 >>> >>>  from bboxes >>> bboxes =  300, 500, 10, 15], [600, 700, 20, 15 >>> >>>  from anchor and roi >>> spots =  400, 600, torch.zeros 10, 15 ], [700, 800, torch.zeros 20, 15  >>> >>>  union of the spots >>> diagram = BaseDiagram(get_sample( >>> diagram.set_spots(diagram_ref, bboxes, spots) >>> len(diagram) 14 >>>",
"func":1
},
{
"ref":"laueimproc.classes.diagram.Diagram.state",
"url":1,
"doc":"Return a hash of the diagram. If two diagrams gots the same state, it means they are the same. The hash take in consideration the internal state of the diagram. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.state '9570d3b8743757aa3bf89a42bc4911de' >>> diagram.find_spots() >>> diagram.state '539428b799f2640150aab633de49b695' >>>"
},
{
"ref":"laueimproc.classes.dataset",
"url":22,
"doc":"Link serveral diagrams together."
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset",
"url":22,
"doc":"Link Diagrams together. Initialise the dataset. Parameters       diagram_refs : tuple The diagram references, transmitted to  add_diagrams . diag2ind : callable, default=laueimproc.classes.base_dataset.default_diag2ind The function to associate an index to a diagram. If provided, this function has to be pickleable. It has to take only one positional argument (a simple Diagram instance) and to return a positive integer. diag2scalars : callable, optional If provided, it allows you to select diagram from physical parameters. \\(f:I\\to\\mathbb{R}^n\\), an injective application. \\(I\\subset\\mathbb{N}\\) is the set of the indices of the diagrams in the dataset. \\(n\\) corresponds to the number of physical parameters. For example the beam positon on the sample or, more generally, a vector of scalar parameters like time, voltage, temperature, pressure . The function must match  laueimproc.ml.dataset_dist.check_diag2scalars_typing ."
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.compute_mean_image",
"url":22,
"doc":"Compute the average image. Based on  laueimproc.immix.mean.mean_stack . Returns    - torch.Tensor The average image of all the images contained in this dataset.",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.compute_inter_image",
"url":22,
"doc":"Compute the median, first quartile, third quartile or everything in between. Parameters       args,  kwargs Transmitted to  laueimproc.immix.inter.sort_stack or  laueimproc.immix.inter.snowflake_stack . method : str The algorithm used. It can be  sort for the naive accurate algorithm, or  snowflake for a faster algorithm on big dataset, but less accurate. Returns    - torch.Tensor The 2d float32 grayscale image.",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.select_closest",
"url":22,
"doc":"Select the closest diagram to a given set of phisical parameters. Parameters      point, tol, scale Transmitted to  laueimproc.ml.dataset_dist.select_closest . no_raise : boolean, default = False If set to True, return None rather throwing a LookupError exception. Returns    - closet_diag :  laueimproc.classes.diagram.Diagram The closet diagram. Examples     >>> from laueimproc.classes.dataset import DiagramsDataset >>> from laueimproc.io import get_samples >>> def position(index: int) -> tuple[int, int]:  . return divmod(index, 10)  . >>> dataset = DiagramsDataset(get_samples(), diag2scalars=position) >>> dataset.select_closest 4.1, 4.9), tol=(0.2, 0.2 Diagram(img_45.jp2) >>>",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.select_closests",
"url":22,
"doc":"Select the diagrams matching a given interval of phisical parameters. Parameters      point, tol, scale Transmitted to  laueimproc.ml.dataset_dist.select_closests . Returns    - sub_dataset :  laueimproc.classes.dataset.DiagramsDataset The frozen view of this dataset containing only the diagram matching the constraints. Examples     >>> from laueimproc.classes.dataset import DiagramsDataset >>> from laueimproc.io import get_samples >>> def position(index: int) -> tuple[int, int]:  . return divmod(index, 10)  . >>> dataset = DiagramsDataset(get_samples(), diag2scalars=position) >>> dataset.select_closests 4.1, 4.9), tol=(1.2, 1.2  >>>",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.train_spot_classifier",
"url":22,
"doc":"Train a variational autoencoder classifier with the spots in the diagrams. It is a non supervised neuronal network. Parameters      model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier, optional If provided, this model will be trained again and returned. shape : float or tuple[int, int], default=0.95 The model input spot shape in numpy convention (height, width). If a percentage is given, the shape is founded with  laueimproc.nn.loader.find_shape . space : float, default = 3.0 The non penalized spreading area half size. Transmitted to  laueimproc.nn.vae_spot_classifier.VAESpotClassifier . intensity_sensitive, scale_sensitive : boolean, default=True Transmitted to  laueimproc.nn.vae_spot_classifier.VAESpotClassifier . epoch, lr, fig Transmitted to  laueimproc.nn.train.train_vae_spot_classifier . Returns    - classifier: laueimproc.nn.vae_spot_classifier.VAESpotClassifier The trained model. Examples     >>> import laueimproc >>> def init(diagram: laueimproc.Diagram):  . diagram.find_spots()  . diagram.filter_spots(range(10  to reduce the number of spots  . >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples( >>> _ = dataset.apply(init) >>> model = dataset.train_spot_classifier(epoch=2) >>>",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.add_property",
"url":2,
"doc":"Add a property to the dataset. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the dataset state changed.",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.add_diagram",
"url":2,
"doc":"Append a new diagram into the dataset. Parameters      new_diagram : laueimproc.classes.diagram.Diagram The new instanciated diagram not already present in the dataset. Raises    LookupError If the diagram is already present in the dataset.",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.add_diagrams",
"url":2,
"doc":"Append the new diagrams into the datset. Parameters      new_diagrams The diagram references, they can be of this natures:  laueimproc.classes.diagram.Diagram : Could be a simple Diagram instance.  iterable : An iterable of any of the types specified above. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> file = min(get_samples().iterdir( >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(Diagram(file  from Diagram instance >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(file)  from filename (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(file  from filename (str) >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(get_samples(  from folder (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(get_samples( )  from folder (str) >>> dataset  >>> >>>  from iterable (DiagramDataset, list, tuple,  .) >>>  ok with nested iterables >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams([Diagram(f) for f in get_samples().iterdir()]) >>> dataset  >>>",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.apply",
"url":2,
"doc":"Apply an operation in all the diagrams of the dataset. Parameters      func : callable A function that take a diagram and optionaly other parameters, and return anything. The function can modify the diagram inplace. It has to be pickaleable. args : tuple, optional Positional arguments transmitted to the provided func. kwargs : dict, optional Keyword arguments transmitted to the provided func. Returns    - res : dict The result of the function for each diagrams. To each diagram index, associate the result. Notes   - This function will be automaticaly applided on the new diagrams, but the result is throw. Examples     >>> from pprint import pprint >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( [ 10]  subset to go faster >>> def peak_search(diagram: Diagram, density: float) -> int:  .  'Return the number of spots. '  . diagram.find_spots(density=density)  . return len(diagram)  . >>> res = dataset.apply(peak_search, args=(0.5, >>> pprint(res) {0: 204, 10: 547, 20: 551, 30: 477, 40: 404, 50: 537, 60: 2121, 70: 274, 80: 271, 90: 481} >>>",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.autosave",
"url":2,
"doc":"Manage the dataset recovery. This allows the dataset to be backuped at regular time intervals. Parameters      filename : pathlike The name of the persistant file, transmitted to  laueimproc.io.save_dataset.write_dataset . delay : float or str, optional If provided and > 0, automatic checkpoint will be performed. it corresponds to the time interval between two check points. The supported formats are defined in  laueimproc.common.time2sec .",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.clone",
"url":2,
"doc":"Instanciate a new identical dataset. Parameters       kwargs : dict Transmitted to  laueimproc.classes.base_diagram.BaseDiagram.clone . Returns    - BaseDiagramsDataset The new copy of self. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset_bis = dataset.clone() >>> assert id(dataset) != id(dataset_bis) >>> assert dataset.state  dataset_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.flush",
"url":2,
"doc":"Extract finished thread diagrams.",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.get_property",
"url":2,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> dataset.add_property(\"prop2\", value=\"any python object 2\") >>> dataset.get_property(\"prop1\") 'any python object 1' >>> dataset.get_property(\"prop2\") 'any python object 2' >>> dataset = dataset[:1]  change state >>> dataset.get_property(\"prop1\") 'any python object 1' >>> try:  . dataset.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the dataset has changed\" >>> try:  . dataset.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.get_scalars",
"url":2,
"doc":"Return the scalars values of each diagrams given by  diag2scalars .",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.indices",
"url":2,
"doc":"Return the sorted map diagram indices currently reachable in the dataset. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset[-10:10:-5].indices [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90] >>>"
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.state",
"url":2,
"doc":"Return a hash of the dataset. If two datasets gots the same state, it means they are the same. The hash take in consideration the indices of the diagrams and the functions applyed. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:].state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:1].state '1de1605a297bafd22a886de7058cae81' >>>"
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.restore",
"url":2,
"doc":"Restore the dataset content from the backup file. Do nothing if the file doesn't exists. Based on  laueimproc.io.save_dataset.restore_dataset . Parameters      filename : pathlike The name of the persistant file.",
"func":1
},
{
"ref":"laueimproc.classes.dataset.DiagramsDataset.run",
"url":2,
"doc":"Run asynchronousely in a child thread, called by self.start().",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset",
"url":2,
"doc":"Define the pytonic structure of a basic BaseDiagramsDataset."
},
{
"ref":"laueimproc.classes.base_dataset.default_diag2ind",
"url":2,
"doc":"General function to find the index of a diagram from the filename. Parameters      diagram : laueimproc.classes.diagram.Diagram The diagram to index. Returns    - index : int The int index written in the file name.",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset",
"url":2,
"doc":"A Basic diagrams dataset with the fondamental structure. Attributes      indices : list[int] The sorted map diagram indices currently reachable in the dataset. Initialise the dataset. Parameters       diagram_refs : tuple The diagram references, transmitted to  add_diagrams . diag2ind : callable, default=laueimproc.classes.base_dataset.default_diag2ind The function to associate an index to a diagram. If provided, this function has to be pickleable. It has to take only one positional argument (a simple Diagram instance) and to return a positive integer. diag2scalars : callable, optional If provided, it allows you to select diagram from physical parameters. \\(f:I\\to\\mathbb{R}^n\\), an injective application. \\(I\\subset\\mathbb{N}\\) is the set of the indices of the diagrams in the dataset. \\(n\\) corresponds to the number of physical parameters. For example the beam positon on the sample or, more generally, a vector of scalar parameters like time, voltage, temperature, pressure . The function must match  laueimproc.ml.dataset_dist.check_diag2scalars_typing ."
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.add_property",
"url":2,
"doc":"Add a property to the dataset. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the dataset state changed.",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.add_diagram",
"url":2,
"doc":"Append a new diagram into the dataset. Parameters      new_diagram : laueimproc.classes.diagram.Diagram The new instanciated diagram not already present in the dataset. Raises    LookupError If the diagram is already present in the dataset.",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.add_diagrams",
"url":2,
"doc":"Append the new diagrams into the datset. Parameters      new_diagrams The diagram references, they can be of this natures:  laueimproc.classes.diagram.Diagram : Could be a simple Diagram instance.  iterable : An iterable of any of the types specified above. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> file = min(get_samples().iterdir( >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(Diagram(file  from Diagram instance >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(file)  from filename (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(file  from filename (str) >>> dataset  >>> >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(get_samples(  from folder (pathlib) >>> dataset  >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams(str(get_samples( )  from folder (str) >>> dataset  >>> >>>  from iterable (DiagramDataset, list, tuple,  .) >>>  ok with nested iterables >>> dataset = BaseDiagramsDataset() >>> dataset.add_diagrams([Diagram(f) for f in get_samples().iterdir()]) >>> dataset  >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.apply",
"url":2,
"doc":"Apply an operation in all the diagrams of the dataset. Parameters      func : callable A function that take a diagram and optionaly other parameters, and return anything. The function can modify the diagram inplace. It has to be pickaleable. args : tuple, optional Positional arguments transmitted to the provided func. kwargs : dict, optional Keyword arguments transmitted to the provided func. Returns    - res : dict The result of the function for each diagrams. To each diagram index, associate the result. Notes   - This function will be automaticaly applided on the new diagrams, but the result is throw. Examples     >>> from pprint import pprint >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.classes.diagram import Diagram >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( [ 10]  subset to go faster >>> def peak_search(diagram: Diagram, density: float) -> int:  .  'Return the number of spots. '  . diagram.find_spots(density=density)  . return len(diagram)  . >>> res = dataset.apply(peak_search, args=(0.5, >>> pprint(res) {0: 204, 10: 547, 20: 551, 30: 477, 40: 404, 50: 537, 60: 2121, 70: 274, 80: 271, 90: 481} >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.autosave",
"url":2,
"doc":"Manage the dataset recovery. This allows the dataset to be backuped at regular time intervals. Parameters      filename : pathlike The name of the persistant file, transmitted to  laueimproc.io.save_dataset.write_dataset . delay : float or str, optional If provided and > 0, automatic checkpoint will be performed. it corresponds to the time interval between two check points. The supported formats are defined in  laueimproc.common.time2sec .",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.clone",
"url":2,
"doc":"Instanciate a new identical dataset. Parameters       kwargs : dict Transmitted to  laueimproc.classes.base_diagram.BaseDiagram.clone . Returns    - BaseDiagramsDataset The new copy of self. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset_bis = dataset.clone() >>> assert id(dataset) != id(dataset_bis) >>> assert dataset.state  dataset_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.flush",
"url":2,
"doc":"Extract finished thread diagrams.",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.get_property",
"url":2,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> dataset.add_property(\"prop2\", value=\"any python object 2\") >>> dataset.get_property(\"prop1\") 'any python object 1' >>> dataset.get_property(\"prop2\") 'any python object 2' >>> dataset = dataset[:1]  change state >>> dataset.get_property(\"prop1\") 'any python object 1' >>> try:  . dataset.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the dataset has changed\" >>> try:  . dataset.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.get_scalars",
"url":2,
"doc":"Return the scalars values of each diagrams given by  diag2scalars .",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.indices",
"url":2,
"doc":"Return the sorted map diagram indices currently reachable in the dataset. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset[-10:10:-5].indices [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90] >>>"
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.state",
"url":2,
"doc":"Return a hash of the dataset. If two datasets gots the same state, it means they are the same. The hash take in consideration the indices of the diagrams and the functions applyed. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_dataset import BaseDiagramsDataset >>> from laueimproc.io import get_samples >>> dataset = BaseDiagramsDataset(get_samples( >>> dataset.state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:].state '047e5d6c00850898c233128e31e1f7e1' >>> dataset[:1].state '1de1605a297bafd22a886de7058cae81' >>>"
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.restore",
"url":2,
"doc":"Restore the dataset content from the backup file. Do nothing if the file doesn't exists. Based on  laueimproc.io.save_dataset.restore_dataset . Parameters      filename : pathlike The name of the persistant file.",
"func":1
},
{
"ref":"laueimproc.classes.base_dataset.BaseDiagramsDataset.run",
"url":2,
"doc":"Run asynchronousely in a child thread, called by self.start().",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram",
"url":1,
"doc":"Define the pytonic structure of a basic Diagram."
},
{
"ref":"laueimproc.classes.base_diagram.check_init",
"url":1,
"doc":"Decorate a Diagram method to ensure the diagram has been init.",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram",
"url":1,
"doc":"A Basic diagram with the fondamental structure. Attributes      areas : torch.Tensor The int32 area of each bboxes. Return None until spots are initialized. bboxes : torch.Tensor or None The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4) (readonly). Return None until spots are initialized. file : pathlib.Path or None The absolute file path to the image if provided, None otherwise (readonly). history : list[str] The actions performed on the Diagram from the initialisation (readonly). image : torch.Tensor The complete brut image of the diagram (readonly). rawrois : torch.Tensor or None The tensor of the raw regions of interest for each spots (readonly). Return None until spots are initialized. The shape is (n, h, w). Contrary to  self.rois , it is only a view of  self.image . rois : torch.Tensor or None The tensor of the provided regions of interest for each spots (readonly). For writing, use  self.find_spost( .) or  self.set_spots( .) . Return None until spots are initialized. The shape is (n, h, w). Create a new diagram with appropriated metadata. Parameters      data : pathlike or arraylike The filename or the array/tensor use as a diagram. For memory management, it is better to provide a pathlike rather than an array."
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.add_property",
"url":1,
"doc":"Add a property to the diagram. Parameters      name : str The identifiant of the property for the requests. If the property is already defined with the same name, the new one erase the older one. value The property value. If a number is provided, it will be faster. erasable : boolean, default=True If set to False, the property will be set in stone, overwise, the property will desappear as soon as the diagram state changed.",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.areas",
"url":1,
"doc":"Return the int32 area of each bboxes. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.areas) None >>> diagram.find_spots() >>> print(diagram.areas)  doctest: +ELLIPSIS tensor([289, 36, 33, 20, 40, 81, 49, 15, 36, 25, 36, 110, 49, 56, 64, 56, 72, 56, 90, 143, 64, 42, 36, 25, 169, 110, 81, 64, 100, 49, 36, 42, 121, 36, 36, 121, 81, 56, 72, 80, 110, 56,  ., 100, 25, 225, 182, 72, 156, 90, 25, 320, 288, 144, 143, 240, 208, 64, 81, 25, 25, 144, 323, 300, 90, 144, 240, 270, 168, 352, 270, 210, 456], dtype=torch.int32) >>>"
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.bboxes",
"url":1,
"doc":"Return the tensor of the bounding boxes (anchor_i, anchor_j, height, width). Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> print(diagram.bboxes) None >>> diagram.find_spots() >>> print(diagram.bboxes)  doctest: +ELLIPSIS tensor( 0, 0, 17, 17], [ 0, 20, 3, 12], [ 0, 1949, 3, 11],  ., [1903, 180, 18, 15], [1930, 1967, 15, 14], [1963, 1170, 24, 19 , dtype=torch.int16) >>>"
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.clone",
"url":1,
"doc":"Instanciate a new identical diagram. Parameters      deep : boolean, default=True If True, the memory of the new created diagram object is totally independ of this object (slow but safe). Otherwise,  self.image ,  self.rois ,  properties and  cached data share the same memory view (Tensor). So modifying one of these attributes in one diagram will modify the same attribute in the other diagram. It if realy faster but not safe. cache : boolean, default=True Copy the cache into the new diagram if True (default), or live the cache empty if False. Returns    - Diagram The new copy of self. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram_bis = diagram.clone() >>> assert id(diagram) != id(diagram_bis) >>> assert diagram.state  diagram_bis.state >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.compress",
"url":1,
"doc":"Delete attributes and elements in the cache. Paremeters      size : int The quantity of bytes to remove from the cache. Returns    - removed : int The number of bytes removed from the cache.",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.file",
"url":1,
"doc":"Return the absolute file path to the image, if it is provided. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> BaseDiagram(get_sample( .file  doctest: +ELLIPSIS PosixPath('/ ./laueimproc/io/ge.jp2') >>>"
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.filter_spots",
"url":1,
"doc":"Keep only the given spots, delete the rest. This method can be used for filtering or sorting spots. Parameters      criteria : arraylike The list of the indices of the spots to keep (negatives indices are allow), or the boolean vector with True for keeping the spot, False otherwise like a mask. msg : str The message to happend to the history. inplace : boolean, default=True If True, modify the diagram self (no copy) and return a reference to self. If False, first clone the diagram, then apply the selection on the new diagram, It create a checkpoint (real copy) (but it is slowler). Returns    - filtered_diagram : BaseDiagram Return None if inplace is True, or a filtered clone of self otherwise. Examples     >>> from pprint import pprint >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> indices = torch.arange(0, len(diagram), 2) >>> diagram.filter_spots(indices, \"keep even spots\") >>> cond = diagram.bboxes[:, 1]  >> diag_final = diagram.filter_spots(cond, \"keep spots on left\", inplace=False) >>> pprint(diagram.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots'] >>> pprint(diag_final.history) ['240 spots from self.find_spots()', '240 to 120 spots: keep even spots', '120 to 63 spots: keep spots on left'] >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.find_spots",
"url":1,
"doc":"Search all the spots in this diagram, store the result into the  spots attribute. Parameters      kwargs : dict Transmitted to  laueimproc.improc.peaks_search.peaks_search .",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.get_properties",
"url":1,
"doc":"Return all the available properties.",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.get_property",
"url":1,
"doc":"Return the property associated to the given id. Parameters      name : str The name of the property to get. Returns    - property : object The property value set with  add_property . Raises    KeyError Is the property has never been defined or if the state changed. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.add_property(\"prop1\", value=\"any python object 1\", erasable=False) >>> diagram.add_property(\"prop2\", value=\"any python object 2\") >>> diagram.get_property(\"prop1\") 'any python object 1' >>> diagram.get_property(\"prop2\") 'any python object 2' >>> diagram.find_spots()  change state >>> diagram.get_property(\"prop1\") 'any python object 1' >>> try:  . diagram.get_property(\"prop2\")  . except KeyError as err:  . print(err)  . \"the property 'prop2' is no longer valid because the state of the diagram has changed\" >>> try:  . diagram.get_property(\"prop3\")  . except KeyError as err:  . print(err)  . \"the property 'prop3' does no exist\" >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.history",
"url":1,
"doc":"Return the actions performed on the Diagram since the initialisation."
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.is_init",
"url":1,
"doc":"Return True if the diagram has been initialized. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.is_init() False >>> diagram.find_spots() >>> diagram.is_init() True >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.image",
"url":1,
"doc":"Return the complete image of the diagram. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.image.shape torch.Size([2018, 2016]) >>> diagram.image.min() >= 0 tensor(True) >>> diagram.image.max()  >>"
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.plot",
"url":1,
"doc":"Prepare for display the diagram and the spots. Parameters      disp : matplotlib.figure.Figure or matplotlib.axes.Axes The matplotlib figure to complete. vmin : float, optional The minimum intensity ploted. vmax : float, optional The maximum intensity ploted. show_axis: boolean, default=True Display the label and the axis if True. show_bboxes: boolean, default=True Draw all the bounding boxes if True. show_image: boolean, default=True Display the image if True, dont call imshow otherwise. Returns    - matplotlib.axes.Axes Filled graphical widget. Notes   - It doesn't create the figure and call show. Use  self.show() to Display the diagram from scratch.",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.rawrois",
"url":1,
"doc":"Return the tensor of the raw rois of the spots."
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.rois",
"url":1,
"doc":"Return the tensor of the provided rois of the spots. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.find_spots() >>> diagram.rawrois.shape torch.Size([240, 24, 19]) >>> diagram.rois.shape torch.Size([240, 24, 19]) >>> diagram.rois.mean()  >>"
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.set_spots",
"url":1,
"doc":"Set the new spots as the current spots, reset the history and the cache. Paremeters      new_spots : tuple Can be an over diagram of type Diagram. Can be an arraylike of bounding boxes n  (anchor_i, anchor_j, height, width). Can be an anchor and a picture n  (anchor_i, anchor_j, roi). It can also be a combination of the above elements to unite the spots. Examples     >>> import torch >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> >>>  from diagram >>> diagram_ref = BaseDiagram(get_sample( >>> diagram_ref.find_spots() >>> diagram_ref.filter_spots(range(10 >>> >>>  from bboxes >>> bboxes =  300, 500, 10, 15], [600, 700, 20, 15 >>> >>>  from anchor and roi >>> spots =  400, 600, torch.zeros 10, 15 ], [700, 800, torch.zeros 20, 15  >>> >>>  union of the spots >>> diagram = BaseDiagram(get_sample( >>> diagram.set_spots(diagram_ref, bboxes, spots) >>> len(diagram) 14 >>>",
"func":1
},
{
"ref":"laueimproc.classes.base_diagram.BaseDiagram.state",
"url":1,
"doc":"Return a hash of the diagram. If two diagrams gots the same state, it means they are the same. The hash take in consideration the internal state of the diagram. The retruned value is a hexadecimal string of length 32. Examples     >>> from laueimproc.classes.base_diagram import BaseDiagram >>> from laueimproc.io import get_sample >>> diagram = BaseDiagram(get_sample( >>> diagram.state '9570d3b8743757aa3bf89a42bc4911de' >>> diagram.find_spots() >>> diagram.state '539428b799f2640150aab633de49b695' >>>"
},
{
"ref":"laueimproc.opti",
"url":23,
"doc":"Provide various tools for improving the efficiency of laueimproc."
},
{
"ref":"laueimproc.opti.collect",
"url":23,
"doc":"Release all unreachable diagrams. Returns    - nbr : int The number of diagrams juste released.",
"func":1
},
{
"ref":"laueimproc.opti.CacheManager",
"url":23,
"doc":"Manage a group of diagram asynchronousely. Parameters      verbose : boolean The chatting status of the experiment (read and write). max_mem_percent : int The maximum amount of memory percent before trying to release some cache (read and write). By default it is based on swapiness. This constructor should always be called with keyword arguments. Arguments are:  group should be None; reserved for future extension when a ThreadGroup class is implemented.  target is the callable object to be invoked by the run() method. Defaults to None, meaning nothing is called.  name is the thread name. By default, a unique name is constructed of the form \"Thread-N\" where N is a small decimal number.  args is a list or tuple of arguments for the target invocation. Defaults to ().  kwargs is a dictionary of keyword arguments for the target invocation. Defaults to {}. If a subclass overrides the constructor, it must make sure to invoke the base class constructor (Thread.__init__( before doing anything else to the thread."
},
{
"ref":"laueimproc.opti.CacheManager.instances",
"url":23,
"doc":""
},
{
"ref":"laueimproc.opti.CacheManager.collect",
"url":23,
"doc":"Try to keep only reachable diagrams.",
"func":1
},
{
"ref":"laueimproc.opti.CacheManager.max_mem_percent",
"url":23,
"doc":"Return the threashold of ram in percent."
},
{
"ref":"laueimproc.opti.CacheManager.run",
"url":23,
"doc":"Asynchron control loop.",
"func":1
},
{
"ref":"laueimproc.opti.CacheManager.track",
"url":23,
"doc":"Track the new diagram if it is not already tracked.",
"func":1
},
{
"ref":"laueimproc.opti.CacheManager.verbose",
"url":23,
"doc":"Get the chatting status of the experiment."
},
{
"ref":"laueimproc.opti.singleton",
"url":24,
"doc":"Allow to create only one instance of an object."
},
{
"ref":"laueimproc.opti.singleton.MetaSingleton",
"url":24,
"doc":"For share memory inside the current session. If a class inerits from this metaclass, only one instance of the class can exists. If we try to create a new instnce, it will return a reference to the unique instance."
},
{
"ref":"laueimproc.opti.singleton.MetaSingleton.instances",
"url":24,
"doc":""
},
{
"ref":"laueimproc.opti.memory",
"url":25,
"doc":"Help to manage the memory."
},
{
"ref":"laueimproc.opti.memory.free_malloc",
"url":25,
"doc":"Clear the allocated malloc on linux.",
"func":1
},
{
"ref":"laueimproc.opti.memory.get_swappiness",
"url":25,
"doc":"Return the system swapiness value.",
"func":1
},
{
"ref":"laueimproc.opti.memory.mem_to_free",
"url":25,
"doc":"Return the number of bytes to be removed from the cache. Parameters      max_mem_percent : int The maximum percent limit of memory to not excess. Value are in [0, 100]. Returns    - mem_to_free : int The amount of memory to freed in order to reach the threshold",
"func":1
},
{
"ref":"laueimproc.opti.memory.total_memory",
"url":25,
"doc":"Return the total usable memory in bytes.",
"func":1
},
{
"ref":"laueimproc.opti.memory.used_memory",
"url":25,
"doc":"Return the total memory used in bytes.",
"func":1
},
{
"ref":"laueimproc.opti.rois",
"url":26,
"doc":"Manage a compact representation of the rois."
},
{
"ref":"laueimproc.opti.rois.filter_by_indices",
"url":26,
"doc":"Select the rois of the given indices. Parameters      indices : torch.Tensor The 1d int64 list of the rois index to keep. Negative indexing is allow. data : bytearray The raw data of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - filtered_data : bytearray The new flatten rois sorted according the provided indices. filtered_bboxes : bytearray The new reorganized bboxes. Notes   - When using the c backend, the negative indices of  indices are set inplace as positive value. Examples     >>> import numpy as np >>> import torch >>> from laueimproc.opti.rois import filter_by_indices >>> indices = torch.tensor([1, 1, 0, -1, -2,  range(100, 1000)]) >>> bboxes = torch.zeros 1000, 4), dtype=torch.int16) >>> bboxes[ 2, 2], bboxes[1 2, 2], bboxes[ 2, 3], bboxes[1 2, 3] = 10, 20, 30, 40 >>> data = bytearray(  . np.linspace(0, 1, (bboxes[:, 2] bboxes[:, 3]).sum(), dtype=np.float32).tobytes()  . ) >>> new_data, new_bboxes = filter_by_indices(indices, data, bboxes) >>> new_bboxes.shape torch.Size([905, 4]) >>> assert new_data  filter_by_indices(indices, data, bboxes, _no_c=True)[0] >>> assert torch.all(new_bboxes  filter_by_indices(indices, data, bboxes, _no_c=True)[1]) >>>",
"func":1
},
{
"ref":"laueimproc.opti.rois.imgbboxes2raw",
"url":26,
"doc":"Extract the rois from the image. Parameters      img : torch.Tensor The float32 grayscale image of a laue diagram of shape (h, w). It doesn't have to be c contiguous but it is faster if it is the case. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. Returns    - data : bytearray The raw data of the concatenated not padded float32 rois. Examples     >>> import torch >>> from laueimproc.opti.rois import imgbboxes2raw >>> img = torch.rand 2000, 2000 >>> bboxes = torch.zeros 1000, 4), dtype=torch.int16) >>> bboxes[ 2, 2], bboxes[1 2, 2], bboxes[ 2, 3], bboxes[1 2, 3] = 10, 20, 30, 40 >>> data = imgbboxes2raw(img, bboxes) >>> len(data) 2200000 >>> assert data  imgbboxes2raw(img, bboxes, _no_c=True) >>>",
"func":1
},
{
"ref":"laueimproc.opti.rois.rawshapes2rois",
"url":26,
"doc":"Unfold and pad the flatten rois data into a tensor. Parameters      data : bytearray The raw data of the concatenated not padded float32 rois. shapes : torch.Tensor The int16 tensor that contains the information of the bboxes shapes. heights = shapes[:, 0] and widths = shapes[:, 1]. It doesn't have to be c contiguous. Returns    - rois : torch.Tensor The unfolded and padded rois of dtype torch.float32 and shape (n, h, w). Examples     >>> import numpy as np >>> import torch >>> from laueimproc.opti.rois import rawshapes2rois >>> shapes = torch.zeros 1000, 2), dtype=torch.int16) >>> shapes[ 2, 0], shapes[1 2, 0], shapes[ 2, 1], shapes[1 2, 1] = 10, 20, 30, 40 >>> data = bytearray(  . np.linspace(0, 1, (shapes[:, 0] shapes[:, 1]).sum(), dtype=np.float32).tobytes()  . ) >>> rois = rawshapes2rois(data, shapes) >>> rois.shape torch.Size([1000, 20, 40]) >>> assert np.array_equal(rois, rawshapes2rois(data, shapes, _no_c=True >>>",
"func":1
},
{
"ref":"laueimproc.opti.rois.roisshapes2raw",
"url":26,
"doc":"Compress the rois into a flatten no padded respresentation. Parameters      rois : torch.Tensor The unfolded and padded rois of dtype torch.float32 and shape (n, h, w). shapes : torch.Tensor The int16 tensor that contains the information of the bboxes shapes. heights = shapes[:, 0] and widths = shapes[:, 1]. It doesn't have to be c contiguous. Returns    - data : bytearray The raw data of the concatenated not padded float32 rois. Examples     >>> import torch >>> from laueimproc.opti.rois import roisshapes2raw >>> shapes = torch.zeros 1000, 2), dtype=torch.int16) >>> shapes[ 2, 0], shapes[1 2, 0], shapes[ 2, 1], shapes[1 2, 1] = 10, 20, 30, 40 >>> rois = torch.zeros 1000, 20, 40), dtype=torch.float32) >>> for i, (h, w) in enumerate(shapes.tolist( :  . rois[i, :h, :w] = (i+1)/1000  . >>> data = roisshapes2raw(rois, shapes) >>> len(data) 2200000 >>> assert data  roisshapes2raw(rois, shapes, _no_c=True) >>>",
"func":1
},
{
"ref":"laueimproc.opti.gpu",
"url":27,
"doc":"Manage the auto gpu."
},
{
"ref":"laueimproc.opti.gpu.to_device",
"url":27,
"doc":"Trensfer all tensors to the new device.",
"func":1
},
{
"ref":"laueimproc.opti.cache",
"url":28,
"doc":"Help to manage the cache."
},
{
"ref":"laueimproc.opti.cache.CacheManager",
"url":28,
"doc":"Manage a group of diagram asynchronousely. Parameters      verbose : boolean The chatting status of the experiment (read and write). max_mem_percent : int The maximum amount of memory percent before trying to release some cache (read and write). By default it is based on swapiness. This constructor should always be called with keyword arguments. Arguments are:  group should be None; reserved for future extension when a ThreadGroup class is implemented.  target is the callable object to be invoked by the run() method. Defaults to None, meaning nothing is called.  name is the thread name. By default, a unique name is constructed of the form \"Thread-N\" where N is a small decimal number.  args is a list or tuple of arguments for the target invocation. Defaults to ().  kwargs is a dictionary of keyword arguments for the target invocation. Defaults to {}. If a subclass overrides the constructor, it must make sure to invoke the base class constructor (Thread.__init__( before doing anything else to the thread."
},
{
"ref":"laueimproc.opti.cache.CacheManager.instances",
"url":28,
"doc":""
},
{
"ref":"laueimproc.opti.cache.CacheManager.collect",
"url":28,
"doc":"Try to keep only reachable diagrams.",
"func":1
},
{
"ref":"laueimproc.opti.cache.CacheManager.max_mem_percent",
"url":28,
"doc":"Return the threashold of ram in percent."
},
{
"ref":"laueimproc.opti.cache.CacheManager.run",
"url":28,
"doc":"Asynchron control loop.",
"func":1
},
{
"ref":"laueimproc.opti.cache.CacheManager.track",
"url":28,
"doc":"Track the new diagram if it is not already tracked.",
"func":1
},
{
"ref":"laueimproc.opti.cache.CacheManager.verbose",
"url":28,
"doc":"Get the chatting status of the experiment."
},
{
"ref":"laueimproc.opti.cache.auto_cache",
"url":28,
"doc":"Decorate to manage the cache of a Diagram method.",
"func":1
},
{
"ref":"laueimproc.opti.cache.getsizeof",
"url":28,
"doc":"Recursive version of sys.getsizeof.",
"func":1
},
{
"ref":"laueimproc.opti.cache.collect",
"url":28,
"doc":"Release all unreachable diagrams. Returns    - nbr : int The number of diagrams juste released.",
"func":1
},
{
"ref":"laueimproc.gmm",
"url":29,
"doc":"Gaussian mixture model. Scalar Terminology           \\(D\\): The space dimension, same as the number of variables. Index \\(d \\in [\\![1;D]\\!]\\).  \\(K\\): The number of gaussians. Index \\(j \\in [\\![1;K]\\!]\\).  \\(N\\): The number of observations. Index \\(i \\in [\\![1;N]\\!]\\). Tensor Terminology           \\( \\mathbf{\\mu}_j = \\begin{pmatrix} \\mu_1  \\vdots  \\mu_D  \\end{pmatrix}_j \\): The center of the gaussian \\(j\\).  \\( \\mathbf{\\Sigma}_j = \\begin{pmatrix} \\sigma_{1,1} & \\dots & \\sigma_{1,D}  \\vdots & \\ddots & \\vdots  \\sigma_{D,1} & \\dots & \\sigma_{d,D}  \\end{pmatrix}_j \\): The full symetric positive covariance matrix of the gaussian \\(j\\).  \\(\\eta_j\\), the relative mass of the gaussian \\(j\\). We have \\(\\sum\\limits_{j=1}^K \\eta_j = 1\\).  \\(\\alpha_i\\): The weights has the relative number of time the individual has been drawn.  \\(\\omega_i\\): The weights has the inverse of the relative covariance of each individual.  \\( \\mathbf{x}_i = \\begin{pmatrix} x_1  \\vdots  x_D  \\end{pmatrix}_j \\): The observation \\(i\\). Calculus Terminology            \\( \\mathcal{N}_{\\mathbf{\\mu}_j, \\mathbf{\\Sigma}_j}(\\mathbf{x}_i) = \\frac { e^{ -\\frac{1}{2} (\\mathbf{x}_i-\\mathbf{\\mu}_j)^\\intercal \\mathbf{\\Sigma}_j^{-1} (\\mathbf{x}_i-\\mathbf{\\mu}_j) } } {\\sqrt{(2\\pi)^D |\\mathbf{\\Sigma}_j}|} \\): The multidimensional gaussian probability density.  \\( \\Gamma(\\mathbf{x}_i) = \\sum\\limits_{j=1}^K \\eta_j \\mathcal{N}_{\\mathbf{\\mu}_j, \\mathbf{\\Sigma}_j}(\\mathbf{x}_i) \\): The total probability density of the observation \\(\\mathbf{x}_i\\)."
},
{
"ref":"laueimproc.gmm.check",
"url":30,
"doc":"Protection against the user."
},
{
"ref":"laueimproc.gmm.check.check_gmm",
"url":30,
"doc":"Ensure the provided parameters are corrects. Parameters      gmm : tuple of torch.Tensor  mean : torch.Tensor The column mean vector \\(\\mathbf{\\mu}_j\\) of shape ( ., \\(K\\), \\(D\\ .  cov : torch.Tensor The covariance matrix \\(\\mathbf{\\Sigma}\\) of shape ( ., \\(K\\), \\(D\\), \\(D\\ .  eta : torch.Tensor The relative mass \\(\\eta_j\\) of shape ( ., \\(K\\ . Raises    AssertionError If the parameters are not correct.",
"func":1
},
{
"ref":"laueimproc.gmm.check.check_infit",
"url":30,
"doc":"Ensure the provided parameters are corrects. Parameters      obs : torch.Tensor The observations \\(\\mathbf{x}_i\\) of shape ( ., \\(N\\), \\(D\\ . weights : torch.Tensor, optional The duplication weights of shape ( ., \\(N\\ . Raises    AssertionError If the parameters are not correct.",
"func":1
},
{
"ref":"laueimproc.gmm.check.check_ingauss",
"url":30,
"doc":"Ensure the provided parameters are corrects. Parameters      obs : torch.Tensor The observations \\(\\mathbf{x}_i\\) of shape ( ., \\(N\\), \\(D\\ . mean : torch.Tensor The column mean vector \\(\\mathbf{\\mu}_j\\) of shape ( ., \\(K\\), \\(D\\ . cov : torch.Tensor The covariance matrix \\(\\mathbf{\\Sigma}\\) of shape ( ., \\(K\\), \\(D\\), \\(D\\ . Raises    AssertionError If the parameters are not correct.",
"func":1
},
{
"ref":"laueimproc.gmm.gauss",
"url":31,
"doc":"Helper for compute multivariate gaussian."
},
{
"ref":"laueimproc.gmm.gauss.gauss",
"url":31,
"doc":"Compute a multivariate gaussian. \\( \\mathcal{N}_{\\mathbf{\\mu}_j, \\mathbf{\\Sigma}_j}(\\mathbf{x}_i) = \\frac { e^{ -\\frac{1}{2} (\\mathbf{x}_i-\\mathbf{\\mu}_j)^\\intercal \\mathbf{\\Sigma}_j^{-1} (\\mathbf{x}_i-\\mathbf{\\mu}_j) } } {\\sqrt{(2\\pi)^D |\\mathbf{\\Sigma}_j}|} \\) Parameters      obs : torch.Tensor The observations \\(\\mathbf{x}_i\\) of shape ( ., \\(N\\), \\(D\\ . mean : torch.Tensor The column mean vector \\(\\mathbf{\\mu}_j\\) of shape ( ., \\(K\\), \\(D\\), 1). cov : torch.Tensor The covariance matrix \\(\\mathbf{\\Sigma}\\) of shape ( ., \\(K\\), \\(D\\), \\(D\\ . Returns    - prob : torch.Tensor The prob density to draw the sample obs of shape ( ., \\(K\\), \\(N\\ . Examples     >>> import torch >>> from laueimproc.gmm.gauss import gauss >>> obs = torch.randn 1000, 10, 4  ( ., n_obs, n_var) >>> mean = torch.randn 1000, 3, 4  ( ., n_clu, n_var) >>> cov = obs.mT @ obs  create real symmetric positive covariance matrix >>> cov = cov.unsqueeze(-3).expand(1000, 3, 4, 4)  ( ., n_clu, n_var, n_var) >>> >>> prob = gauss(obs, mean, cov) >>> prob.shape torch.Size([1000, 3, 10]) >>> >>> mean.requires_grad = cov.requires_grad = True >>> gauss(obs, mean, cov).sum().backward() >>>",
"func":1
},
{
"ref":"laueimproc.gmm.gauss.gauss2d",
"url":31,
"doc":"Compute a 2d gaussian. Approximatively 30% faster than general  gauss . Parameters      obs : torch.Tensor The observations of shape ( ., \\(N\\), 2). \\(\\mathbf{x}_i = \\begin{pmatrix} x_1  x_2  \\end{pmatrix}_j\\) mean : torch.Tensor The 2x1 column mean vector of shape ( ., \\(K\\), 2). \\(\\mathbf{\\mu}_j = \\begin{pmatrix} \\mu_1  \\mu_2  \\end{pmatrix}_j\\) cov : torch.Tensor The 2x2 covariance matrix of shape ( ., \\(K\\), 2, 2). \\(\\mathbf{\\Sigma} = \\begin{pmatrix} \\sigma_1 & c  c & \\sigma_2  \\end{pmatrix}\\) with \\(\\begin{cases} \\sigma_1 > 0  \\sigma_2 > 0  \\end{cases}\\) Returns    - prob : torch.Tensor The prob density to draw the sample obs of shape ( ., \\(K\\), \\(N\\ . It is associated to \\(\\mathcal{N}_{\\mathbf{\\mu}_j, \\mathbf{\\Sigma}_j}\\). Examples     >>> import torch >>> from laueimproc.gmm.gauss import gauss2d >>> obs = torch.randn 1000, 100, 2  ( ., n_obs, n_var) >>> mean = torch.randn 1000, 3, 2  ( ., n_clu, n_var) >>> cov = obs.mT @ obs  create real symmetric positive covariance matrix >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2)  ( ., n_clu, n_var, n_var) >>> >>> prob = gauss2d(obs, mean, cov) >>> prob.shape torch.Size([1000, 3, 100]) >>> >>> mean.requires_grad = cov.requires_grad = True >>> gauss2d(obs, mean, cov).sum().backward() >>>",
"func":1
},
{
"ref":"laueimproc.gmm.fit",
"url":32,
"doc":"Implement the EM (Esperance Maximisation) algo. Detailed algorithm for going from step \\(s\\) to step \\(s+1\\):  \\( p_{i,j} = \\frac{ \\eta_j^{(s)} \\mathcal{N}_{\\mathbf{\\mu}_j^{(s)}, \\mathbf{\\Sigma}_j^{(s) \\left(\\mathbf{x}_i\\right) }{ \\sum\\limits_{k=1}^K \\eta_k^{(s)} \\mathcal{N}_{\\mathbf{\\mu}_k^{(s)}, \\mathbf{\\Sigma}_k^{(s) \\left(\\mathbf{x}_i\\right) } \\) Posterior probability that observation \\(i\\) belongs to cluster \\(j\\).  \\( \\eta_j^{(s+1)} = \\frac{\\sum\\limits_{i=1}^N \\alpha_i p_{i,j {\\sum\\limits_{i=1}^N \\alpha_i} \\) The relative weight of each gaussian.  \\( \\mathbf{\\mu}_j^{(s+1)} = \\frac{ \\sum\\limits_{i=1}^N \\alpha_i \\omega_i p_{i,j} \\mathbf{x}_i }{ \\sum\\limits_{i=1}^N \\alpha_i \\omega_i p_{i,j} } \\) The mean of each gaussian.  \\( \\mathbf{\\Sigma}_j^{(s+1)} = \\frac{ \\sum\\limits_{i=1}^N \\omega_i \\alpha_i p_{i,j} \\left(\\mathbf{x}_i - \\mathbf{\\mu}_j^{(s+1)}\\right) \\left(\\mathbf{x}_i - \\mathbf{\\mu}_j^{(s+1)}\\right)^{\\intercal} }{ \\sum\\limits_{i=1}^N \\alpha_i p_{i,j} } \\) The cov of each gaussian. This sted is iterated as long as the log likelihood increases."
},
{
"ref":"laueimproc.gmm.fit.fit_em",
"url":32,
"doc":"Implement a weighted version of the 2d EM algorithm. Parameters      roi : torch.Tensor The picture of the roi, of shape (h, w). nbr_clusters : int The number \\(K\\) of gaussians. nbr_tries : int The number of times that the algorithm converges in order to have the best possible solution. It is ignored in the case  nbr_clusters = 1. Returns    - mean : torch.Tensor The vectors \\(\\mathbf{\\mu}\\). Shape ( ., \\(K\\), \\(D\\ . cov : torch.Tensor The matrices \\(\\mathbf{\\Sigma}\\). Shape ( ., \\(K\\), \\(D\\), \\(D\\ . eta : torch.Tensor The relative mass \\(\\eta\\). Shape ( ., \\(K\\ .",
"func":1
},
{
"ref":"laueimproc.gmm.metric",
"url":33,
"doc":"Metric to estimate the quality of the fit of the gmm."
},
{
"ref":"laueimproc.gmm.metric.aic",
"url":33,
"doc":"Compute the Akaike Information Criterion and the Bayesian Information Criterion. Parameters      obs : torch.Tensor The observations \\(\\mathbf{x}_i\\) of shape ( ., \\(N\\), \\(D\\ . weights : torch.Tensor, optional The duplication weights of shape ( ., \\(N\\ . gmm : tuple of torch.Tensor  mean : torch.Tensor The column mean vector \\(\\mathbf{\\mu}_j\\) of shape ( ., \\(K\\), \\(D\\ .  cov : torch.Tensor The covariance matrix \\(\\mathbf{\\Sigma}\\) of shape ( ., \\(K\\), \\(D\\), \\(D\\ .  eta : torch.Tensor The relative mass \\(\\eta_j\\) of shape ( ., \\(K\\ . Returns    - aic : torch.Tensor Akaike Information Criterion \\(aic = 2p-2\\log(L_{\\alpha,\\omega})\\), \\(p\\) is the number of free parameters and \\(L_{\\alpha,\\omega}\\) the likelihood.",
"func":1
},
{
"ref":"laueimproc.gmm.metric.bic",
"url":33,
"doc":"Compute the Akaike Information Criterion and the Bayesian Information Criterion. Parameters      obs : torch.Tensor The observations \\(\\mathbf{x}_i\\) of shape ( ., \\(N\\), \\(D\\ . weights : torch.Tensor, optional The duplication weights of shape ( ., \\(N\\ . gmm : tuple of torch.Tensor  mean : torch.Tensor The column mean vector \\(\\mathbf{\\mu}_j\\) of shape ( ., \\(K\\), \\(D\\ .  cov : torch.Tensor The covariance matrix \\(\\mathbf{\\Sigma}\\) of shape ( ., \\(K\\), \\(D\\), \\(D\\ .  eta : torch.Tensor The relative mass \\(\\eta_j\\) of shape ( ., \\(K\\ . Returns    - bic : torch.Tensor Bayesian Information Criterion \\(bic = \\log(N)p-2\\log(L_{\\alpha,\\omega})\\), \\(p\\) is the number of free parameters and \\(L_{\\alpha,\\omega}\\) the likelihood.",
"func":1
},
{
"ref":"laueimproc.gmm.metric.log_likelihood",
"url":33,
"doc":"Compute the log likelihood. Parameters      obs : torch.Tensor The observations \\(\\mathbf{x}_i\\) of shape ( ., \\(N\\), \\(D\\ . weights : torch.Tensor, optional The duplication weights of shape ( ., \\(N\\ . gmm : tuple of torch.Tensor  mean : torch.Tensor The column mean vector \\(\\mathbf{\\mu}_j\\) of shape ( ., \\(K\\), \\(D\\ .  cov : torch.Tensor The covariance matrix \\(\\mathbf{\\Sigma}\\) of shape ( ., \\(K\\), \\(D\\), \\(D\\ .  eta : torch.Tensor The relative mass \\(\\eta_j\\) of shape ( ., \\(K\\ . Returns    - log_likelihood : torch.Tensor The log likelihood \\( L_{\\alpha,\\omega} \\) of shape ( .,). \\( L_{\\alpha,\\omega} = \\log\\left( \\prod\\limits_{i=1}^N \\sum\\limits_{j=1}^K \\eta_j \\left( \\mathcal{N}_{(\\mathbf{\\mu}_j,\\frac{1}{\\omega_i}\\mathbf{\\Sigma}_j)}(\\mathbf{x}_i) \\right)^{\\alpha_i} \\right) \\)",
"func":1
},
{
"ref":"laueimproc.gmm.metric.mse",
"url":33,
"doc":"Compute the mean square error. Parameters      obs : torch.Tensor The observations \\(\\mathbf{x}_i\\) of shape ( ., \\(N\\), \\(D\\ . weights : torch.Tensor, optional The duplication weights of shape ( ., \\(N\\ . gmm : tuple of torch.Tensor  mean : torch.Tensor The column mean vector \\(\\mathbf{\\mu}_j\\) of shape ( ., \\(K\\), \\(D\\ .  cov : torch.Tensor The covariance matrix \\(\\mathbf{\\Sigma}\\) of shape ( ., \\(K\\), \\(D\\), \\(D\\ .  mag : torch.Tensor The relative mass \\(\\eta_j\\) of shape ( ., \\(K\\ . Returns    - mse : torch.Tensor The mean square error of shape ( .,). \\( mse = \\frac{1}{N}\\sum\\limits_{i=1}^N \\left(\\left(\\sum\\limits_{j=1}^K \\eta_j \\left( \\mathcal{N}_{(\\mathbf{\\mu}_j,\\frac{1}{\\omega_i}\\mathbf{\\Sigma}_j)}(\\mathbf{x}_i) \\right)\\right) - \\alpha_i\\right)^2 \\)",
"func":1
},
{
"ref":"laueimproc.gmm.linalg",
"url":34,
"doc":"Helper for fast linear algebra for 2d matrix."
},
{
"ref":"laueimproc.gmm.linalg.batched_matmul",
"url":34,
"doc":"Perform a matrix product on the last 2 dimensions. Parameters      mat1 : torch.Tensor Matrix of shape ( ., n, m). mat2 : torch.Tensor Matrix of shape ( ., m, p). Returns    - prod : torch.Tensor The matrix of shape ( ., n, p). Examples     >>> import torch >>> from laueimproc.gmm.linalg import batched_matmul >>> mat1 = torch.randn 9, 1, 7, 6, 3, 4 >>> mat2 = torch.randn( (8, 1, 6, 4, 5 >>> batched_matmul(mat1, mat2).shape torch.Size([9, 8, 7, 6, 3, 5]) >>>",
"func":1
},
{
"ref":"laueimproc.gmm.linalg.cov2d_to_eigtheta",
"url":34,
"doc":"Rotate the covariance matrix into canonical base, like a PCA. Work only for dimension \\(D = 2\\), ie 2 2 square matrix. Parameters      cov : torch.Tensor The 2x2 covariance matrix of shape ( ., 2, 2). \\(\\mathbf{\\Sigma} = \\begin{pmatrix} \\sigma_1 & c  c & \\sigma_2  \\end{pmatrix}\\) with \\(\\begin{cases} \\sigma_1 > 0  \\sigma_2 > 0  \\end{cases}\\) eig : boolean, default=True If True, compute the eigen values, leave the field empty otherwise (faster). theta : boolean, default=True If True, compute the eigen vector rotation, leave the field empty otherwise (faster). Returns    - eigtheta : torch.Tensor The concatenation of the eigenvalues and theta in counterclockwise (trigo). \\( \\left[ \\lambda_1, \\lambda_2, \\theta \\right] \\) of shape ( ., 3) such that: \\(\\begin{cases} \\lambda_1 >= \\lambda_2 > 0  \\theta \\in \\left]-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]  \\mathbf{R} = \\begin{pmatrix} cos(\\theta) & -sin(\\theta)  sin(\\theta) & cos(\\theta)  \\end{pmatrix}  \\mathbf{D} = \\begin{pmatrix} \\lambda_1 & 0  0 & \\lambda_2  \\end{pmatrix}  \\mathbf{R}^{-1} \\mathbf{\\Sigma} \\mathbf{R} = \\mathbf{D}  \\end{cases}\\) \\(\\begin{cases} tr( \\mathbf{\\Sigma} ) = tr( \\mathbf{R}^{-1} \\mathbf{\\Sigma} \\mathbf{R} ) = tr(\\mathbf{D})  det( \\mathbf{\\Sigma} ) = det( \\mathbf{R}^{-1} \\mathbf{\\Sigma} \\mathbf{R} ) = det(\\mathbf{D})  \\mathbf{\\Sigma} \\begin{pmatrix} cos(\\theta)  sin(\\theta)  \\end{pmatrix} = \\lambda_1 \\begin{pmatrix} cos(\\theta)  sin(\\theta)  \\end{pmatrix}  \\end{cases}\\) \\( \\Leftrightarrow \\begin{cases} \\lambda_1 + \\lambda_2 = \\sigma_1 + \\sigma_2  \\lambda_1 \\lambda_2 = \\sigma_1 \\sigma_2 - c^2  \\sigma_1 cos(\\theta) + c sin(\\theta) = \\lambda_1 cos(\\theta)  \\end{cases}\\) \\( \\Leftrightarrow \\begin{cases} \\lambda_1 = \\frac{1}{2} \\left( \\sigma_1 + \\sigma_2 + \\sqrt{(2c)^2 + (\\sigma_2 - \\sigma_1)^2} \\right)  \\lambda_2 = \\frac{1}{2} \\left( \\sigma_1 + \\sigma_2 - \\sqrt{(2c)^2 + (\\sigma_2 - \\sigma_1)^2} \\right)  \\theta = tan^{-1}\\left( \\frac{ \\sigma_2 - \\sigma_1 + \\sqrt{(2c)^2 + (\\sigma_2 - \\sigma_1)^2} }{2c} \\right)  \\end{cases}\\) Examples     >>> import numpy as np >>> import torch >>> from laueimproc.gmm.linalg import cov2d_to_eigtheta >>> obs = torch.randn 1000, 100, 2), dtype=torch.float64)  ( ., n_obs, n_var) >>> cov = obs.mT @ obs  create real symmetric positive covariance matrix >>> eigtheta = cov2d_to_eigtheta(cov) >>> >>>  check resultst are corrects >>> torch.allclose(torch.linalg.eigvalsh(cov).flip(-1), eigtheta[ ., :2]) True >>> theta = eigtheta[ ., 2] >>> rot =  torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta) >>> rot = torch.asarray(np.array(rot .movedim 0, -2), (1, -1 >>> diag = torch.zeros_like(cov) >>> diag[ ., 0, 0] = eigtheta[ ., 0] >>> diag[ ., 1, 1] = eigtheta[ ., 1] >>> torch.allclose(torch.linalg.inv(rot) @ cov @ rot, diag) True >>> >>>  check is differenciable >>> cov.requires_grad = True >>> cov2d_to_eigtheta(cov).sum().backward() >>> >>>  timing >>> import timeit >>> def timer():  . a = min(timeit.repeat(lambda: cov2d_to_eigtheta(cov), repeat=10, number=100  . b = min(timeit.repeat(lambda: torch.linalg.eigh(cov), repeat=10, number=100  . print(f\"torch is {b/a:.2f} times slowler\")  . a = min(timeit.repeat(lambda: cov2d_to_eigtheta(cov, eig=False), repeat=10, number=100  . b = min(timeit.repeat(lambda: torch.linalg.eigvalsh(cov), repeat=10, number=100  . print(f\"torch is {b/a:.2f} times slowler\")  . >>> timer()  doctest: +SKIP torch is 2.84 times slowler torch is 3.87 times slowler >>>",
"func":1
},
{
"ref":"laueimproc.gmm.linalg.inv_cov2d",
"url":34,
"doc":"Compute the det and the inverse of covariance matrix. Parameters      cov : torch.Tensor The 2x2 covariance matrix of shape ( ., 2, 2). \\(\\mathbf{\\Sigma} = \\begin{pmatrix} \\sigma_1 & c  c & \\sigma_2  \\end{pmatrix}\\) with \\(\\begin{cases} \\sigma_1 > 0  \\sigma_2 > 0  \\end{cases}\\) inv : boolean, default=True If True, compute the inverse matrix, return empty tensor overwise otherwise (faster). Returns    - det : torch.Tensor The determinant of the matrix of shape ( .,). inv : torch.Tensor The inverse matrix of shape ( ., 2, 2). Examples     >>> import torch >>> from laueimproc.gmm.linalg import inv_cov2d >>> obs = torch.randn 1000, 100, 2  ( ., n_obs, n_var) >>> cov = obs.mT @ obs  create real symmetric positive covariance matrix >>> _, inv = inv_cov2d(cov) >>> torch.allclose(torch.linalg.inv(cov), inv) True >>>",
"func":1
},
{
"ref":"laueimproc.gmm.linalg.multivariate_normal",
"url":34,
"doc":"Draw a random realisations of 2d centered gaussian law. Parameters      cov : torch.Tensor Covariance matrix of shape ( ., 2, 2). nbr : int The numbers of samples by cov matrices. Returns    - draw : torch.Tensor The random centered draw of cov matrix cov, shape (nbr,  ., 2). Examples     >>> import math, torch >>> from laueimproc.gmm.linalg import multivariate_normal >>> >>> s1, s2, t = 9.0, 2.0, math.radians(20) >>> rot = torch.asarray( math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t) ) >>> samples = torch.randn(10_000, 2)  N(0,  1, 0], [0, 1 ) >>> samples  = torch.asarray( s1, s2 )  N(0,  s1 2, 0], [0, s2 2 ) >>> samples @= torch.linalg.inv(rot) >>> cov = (samples.mT @ samples) / (len(samples) - 1) >>> >>> draw = multivariate_normal(cov, len(samples >>> cov_ = (draw.mT @ draw) / (len(draw) - 1) >>> torch.allclose(cov, cov_, rtol=0.1) True >>> >>>  import matplotlib.pyplot as plt >>>  _ = plt.scatter( samples.mT, alpha=0.1) >>>  _ = plt.scatter( draw.mT, alpha=0.1) >>>  _ = plt.axis(\"equal\") >>>  plt.show() >>>",
"func":1
},
{
"ref":"laueimproc.gmm.gmm",
"url":35,
"doc":"Helper for compute a mixture of multivariate gaussians."
},
{
"ref":"laueimproc.gmm.gmm.gmm2d",
"url":35,
"doc":"Compute the weighted sum of several 2d gaussians. Parameters      obs : torch.Tensor The observations of shape ( ., \\(N\\), 2). \\(\\mathbf{x}_i = \\begin{pmatrix} x_1  x_2  \\end{pmatrix}_j\\) mean : torch.Tensor The 2x1 column mean vector of shape ( ., \\(K\\), 2). \\(\\mathbf{\\mu}_j = \\begin{pmatrix} \\mu_1  \\mu_2  \\end{pmatrix}_j\\) cov : torch.Tensor The 2x2 covariance matrix of shape ( ., \\(K\\), 2, 2). \\(\\mathbf{\\Sigma} = \\begin{pmatrix} \\sigma_1 & c  c & \\sigma_2  \\end{pmatrix}\\) with \\(\\begin{cases} \\sigma_1 > 0  \\sigma_2 > 0  \\end{cases}\\) eta : torch.Tensor The scalar mass of each gaussian \\(\\eta_j\\) of shape ( ., \\(K\\ . Returns    - prob_cum The weighted sum of the proba of each gaussian for each observation. The shape of \\(\\Gamma(\\mathbf{x}_i)\\) if ( ., \\(N\\ .",
"func":1
},
{
"ref":"laueimproc.gmm.gmm.gmm2d_and_jac",
"url":35,
"doc":"Compute the grad of a 2d mixture gaussian model. Parameters      obs : torch.Tensor The observations of shape ( ., \\(N\\), 2). \\(\\mathbf{x}_i = \\begin{pmatrix} x_1  x_2  \\end{pmatrix}_j\\) mean : torch.Tensor The 2x1 column mean vector of shape ( ., \\(K\\), 2). \\(\\mathbf{\\mu}_j = \\begin{pmatrix} \\mu_1  \\mu_2  \\end{pmatrix}_j\\) cov : torch.Tensor The 2x2 covariance matrix of shape ( ., \\(K\\), 2, 2). \\(\\mathbf{\\Sigma} = \\begin{pmatrix} \\sigma_1 & c  c & \\sigma_2  \\end{pmatrix}\\) with \\(\\begin{cases} \\sigma_1 > 0  \\sigma_2 > 0  \\end{cases}\\) mag : torch.Tensor The scalar mass of each gaussian \\(\\eta_j\\) of shape ( ., \\(K\\ . Returns    - prob : torch.Tensor Returned value from  gmm2d of shape ( ., \\(N\\ . mean_jac : torch.Tensor The jacobian of the 2x1 column mean vector of shape ( ., \\(N\\), \\(K\\), 2). cov_jac : torch.Tensor The jacobian of the 2x2 half covariance matrix of shape ( ., \\(N\\), \\(K\\), 2, 2). Take care of the factor 2 of the coefficient of correlation. mag_jac : torch.Tensor The jacobian of the scalar mass of each gaussian of shape ( ., \\(N\\), \\(K\\ . Examples     >>> import torch >>> from laueimproc.gmm.gmm import gmm2d_and_jac >>> obs = torch.randn 1000, 100, 2  ( ., n_obs, n_var) >>> mean = torch.randn 1000, 3, 2  ( ., n_clu, n_var) >>> cov = obs.mT @ obs  create real symmetric positive covariance matrix >>> cov = cov.unsqueeze(-3).expand(1000, 3, 2, 2).clone()  ( ., n_clu, n_var, n_var) >>> mag = torch.rand 1000, 3  ( ., n_clu) >>> mag /= mag.sum(dim=-1, keepdim=True) >>> >>> prob, mean_jac, cov_jac, mag_jac = gmm2d_and_jac(obs, mean, cov, mag) >>> prob.shape torch.Size([1000, 100]) >>> mean_jac.shape torch.Size([1000, 100, 3, 2]) >>> cov_jac.shape torch.Size([1000, 100, 3, 2, 2]) >>> mag_jac.shape torch.Size([1000, 100, 3]) >>> >>> prob_, mean_jac_, cov_jac_, mag_jac_ = gmm2d_and_jac(obs, mean, cov, mag, _autodiff=True) >>> assert torch.allclose(prob, prob_) >>> assert torch.allclose(mean_jac, mean_jac_) >>> assert torch.allclose(cov_jac, cov_jac_) >>> assert torch.allclose(mag_jac, mag_jac_) >>>",
"func":1
},
{
"ref":"laueimproc.gmm.gmm.mse_cost",
"url":35,
"doc":"Compute the mse loss between the predicted gmm and the rois. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. mean : torch.Tensor The 2x1 column mean vector of shape (n, \\(K\\), 2). \\(\\mathbf{\\mu}_j = \\begin{pmatrix} \\mu_1  \\mu_2  \\end{pmatrix}_j\\) cov : torch.Tensor The 2x2 covariance matrix of shape (n, \\(K\\), 2, 2). \\(\\mathbf{\\Sigma} = \\begin{pmatrix} \\sigma_1 & c  c & \\sigma_2  \\end{pmatrix}\\) with \\(\\begin{cases} \\sigma_1 > 0  \\sigma_2 > 0  \\end{cases}\\) mag : torch.Tensor The scalar mass of each gaussian \\(\\eta_j\\) of shape (n, \\(K\\ . Returns    - cost : torch.Tensor The value of the reduced loss function evaluated for each roi. The shape in (n,). Examples     >>> import torch >>> from laueimproc.gmm.gmm import mse_cost >>> from laueimproc.opti.rois import roisshapes2raw >>> rois = torch.rand 1000, 10, 10 >>> bboxes = torch.full 1000, 4), 10, dtype=torch.int16) >>> mean = torch.randn 1000, 3, 2 + 15.0  (n, n_clu, n_var, 1) >>> cov = torch.tensor( 1., 0.], [0., 1. ).reshape(1, 1, 2, 2).expand(1000, 3, 2, 2).clone() >>> mag = torch.rand 1000, 3  (n, n_clu) >>> mag /= mag.sum(dim=-1, keepdim=True) >>> data = roisshapes2raw(rois, bboxes[:, 2:]) >>> >>> cost = mse_cost(data, bboxes, mean, cov, mag) >>> cost.shape torch.Size([1000]) >>> >>> cost_ = mse_cost(data, bboxes, mean, cov, mag, _no_c=True) >>> assert torch.allclose(cost, cost_) >>>",
"func":1
},
{
"ref":"laueimproc.gmm.gmm.mse_cost_and_grad",
"url":35,
"doc":"Compute the grad of the loss between the predicted gmm and the rois. Parameters      data : bytearray The raw data \\(\\alpha_i\\) of the concatenated not padded float32 rois. bboxes : torch.Tensor The int16 tensor of the bounding boxes (anchor_i, anchor_j, height, width) for each spots, of shape (n, 4). It doesn't have to be c contiguous. mean : torch.Tensor The 2x1 column mean vector of shape (n, \\(K\\), 2). \\(\\mathbf{\\mu}_j = \\begin{pmatrix} \\mu_1  \\mu_2  \\end{pmatrix}_j\\) cov : torch.Tensor The 2x2 covariance matrix of shape (n, \\(K\\), 2, 2). \\(\\mathbf{\\Sigma} = \\begin{pmatrix} \\sigma_1 & c  c & \\sigma_2  \\end{pmatrix}\\) with \\(\\begin{cases} \\sigma_1 > 0  \\sigma_2 > 0  \\end{cases}\\) mag : torch.Tensor The scalar mass of each gaussian \\(\\eta_j\\) of shape (n, \\(K\\ . Returns    - cost : torch.Tensor The value of the reduced loss function evaluated for each roi. The shape in (n,). mean_grad : torch.Tensor The gradient of the 2x1 column mean vector of shape (n, \\(K\\), 2). cov_grad : torch.Tensor The gradient of the 2x2 half covariance matrix of shape (n, \\(K\\), 2, 2). Take care of the factor 2 of the coefficient of correlation. mag_grad : torch.Tensor The gradient of the scalar mass of each gaussian of shape (n, \\(K\\ . Examples     >>> import torch >>> from laueimproc.gmm.gmm import mse_cost_and_grad >>> from laueimproc.opti.rois import roisshapes2raw >>> rois = torch.rand 1000, 10, 10 >>> bboxes = torch.full 1000, 4), 10, dtype=torch.int16) >>> mean = torch.randn 1000, 3, 2 + 15.0  (n, n_clu, n_var) >>> cov = torch.tensor( 1., 0.], [0., 1. ).reshape(1, 1, 2, 2).expand(1000, 3, 2, 2).clone() >>> mag = torch.rand 1000, 3  (n, n_clu) >>> mag /= mag.sum(dim=-1, keepdim=True) >>> data = roisshapes2raw(rois, bboxes[:, 2:]) >>> >>> cost, mean_grad, cov_grad, mag_grad = mse_cost_and_grad(data, bboxes, mean, cov, mag) >>> tuple(cost.shape), tuple(mean_grad.shape), tuple(cov_grad.shape), tuple(mag_grad.shape)  1000,), (1000, 3, 2), (1000, 3, 2, 2), (1000, 3 >>> >>> cost_, mean_g, cov_g, mag_g = mse_cost_and_grad(data, bboxes, mean, cov, mag, _no_c=True) >>> assert torch.allclose(cost, cost_) >>> assert torch.allclose(mean_grad, mean_g) >>> assert torch.allclose(cov_grad, cov_g) >>> assert torch.allclose(mag_grad, mag_g) >>>",
"func":1
},
{
"ref":"laueimproc.testing",
"url":36,
"doc":"Management of all unit tests."
},
{
"ref":"laueimproc.testing.install",
"url":37,
"doc":"Test if the dependents libraries are well installed and linked. Basicaly, it checks if the installation seems to be correct."
},
{
"ref":"laueimproc.testing.install.test_gpu_torch",
"url":37,
"doc":"Test if torch is able to use the GPU.",
"func":1
},
{
"ref":"laueimproc.testing.tests",
"url":38,
"doc":""
},
{
"ref":"laueimproc.testing.tests.inter_batch",
"url":39,
"doc":"Test the median function."
},
{
"ref":"laueimproc.testing.tests.inter_batch.test_tol_random",
"url":39,
"doc":"Create a random dataset, and compute the median.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots",
"url":40,
"doc":"Test the set_spots function."
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_anchors_rois_numpy",
"url":40,
"doc":"Init a diagram from anchors and rois.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_anchors_rois_torch",
"url":40,
"doc":"Init a diagram from anchors and rois.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_bboxes_list",
"url":40,
"doc":"Init a diagram from bboxes.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_bboxes_numpy",
"url":40,
"doc":"Init a diagram from bboxes.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_bboxes_set",
"url":40,
"doc":"Init a diagram from bboxes.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_bboxes_torch",
"url":40,
"doc":"Init a diagram from bboxes.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_bboxes_tuple",
"url":40,
"doc":"Init a diagram from bboxes.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_from_diagram",
"url":40,
"doc":"Init a diagram from an other.",
"func":1
},
{
"ref":"laueimproc.testing.tests.set_spots.test_reset",
"url":40,
"doc":"Init diagram with empty data.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg",
"url":41,
"doc":"Test the function of bragg diffraction."
},
{
"ref":"laueimproc.testing.tests.bragg.test_batch_hkl_reciprocal_to_energy",
"url":41,
"doc":"Test batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_batch_hkl_reciprocal_to_uq",
"url":41,
"doc":"Test batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_batch_uf_to_uq",
"url":41,
"doc":"Test batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_batch_uq_to_uf",
"url":41,
"doc":"Test batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_bij_uf_to_uq_to_uf",
"url":41,
"doc":"Test uf -> uq -> uf = uf.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_bij_uq_to_uf_to_uq",
"url":41,
"doc":"Test uq -> uf -> uq = uq.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_jac_hkl_reciprocal_to_energy",
"url":41,
"doc":"Test compute jacobian.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_jac_hkl_reciprocal_to_uq",
"url":41,
"doc":"Test compute jacobian.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_normalization_hkl_reciprocal_to_uq",
"url":41,
"doc":"Test norm is 1.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_normalization_uf_to_uq",
"url":41,
"doc":"Test norm is 1.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_normalization_uq_to_uf",
"url":41,
"doc":"Test norm is 1.",
"func":1
},
{
"ref":"laueimproc.testing.tests.bragg.test_sign_uq",
"url":41,
"doc":"Test uf is not sensitive to the uq orientation.",
"func":1
},
{
"ref":"laueimproc.testing.tests.lattice",
"url":42,
"doc":"Test the convertion between lattice and primitive space."
},
{
"ref":"laueimproc.testing.tests.lattice.test_batch_lattice_to_primitive",
"url":42,
"doc":"Tests batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.lattice.test_jac_lattice_to_primitive",
"url":42,
"doc":"Tests compute jacobian.",
"func":1
},
{
"ref":"laueimproc.testing.tests.lattice.test_batch_primitive_to_lattice",
"url":42,
"doc":"Tests batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.lattice.test_jac_primitive_to_lattice",
"url":42,
"doc":"Tests compute jacobian.",
"func":1
},
{
"ref":"laueimproc.testing.tests.lattice.test_bij_lattice_to_primitive_to_lattice",
"url":42,
"doc":"Test is the transformation is reverible.",
"func":1
},
{
"ref":"laueimproc.testing.tests.metric",
"url":43,
"doc":"Test if the matching rate is ok."
},
{
"ref":"laueimproc.testing.tests.metric.test_batch_matching_rate",
"url":43,
"doc":"Test batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.metric.test_batch_matching_rate_continuous",
"url":43,
"doc":"Test batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.metric.test_grad_matching_rate_continuous",
"url":43,
"doc":"Test the continuous matching rate is differenciable.",
"func":1
},
{
"ref":"laueimproc.testing.tests.metric.test_value_matching_rate",
"url":43,
"doc":"Test if the result of the matching rate is correct.",
"func":1
},
{
"ref":"laueimproc.testing.tests.projection",
"url":44,
"doc":"Test the convertion between ray and detector position."
},
{
"ref":"laueimproc.testing.tests.projection.test_batch_detector_to_ray",
"url":44,
"doc":"Tests batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.projection.test_batch_ray_to_detector",
"url":44,
"doc":"Tests batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.projection.test_bij_ray_to_point_to_ray",
"url":44,
"doc":"Test ray -> point -> ray = ray.",
"func":1
},
{
"ref":"laueimproc.testing.tests.projection.test_jac_detector_to_ray",
"url":44,
"doc":"Tests compute jacobian.",
"func":1
},
{
"ref":"laueimproc.testing.tests.projection.test_jac_ray_to_detector",
"url":44,
"doc":"Tests compute jacobian.",
"func":1
},
{
"ref":"laueimproc.testing.tests.projection.test_normalization_detector_to_ray",
"url":44,
"doc":"Test norm is 1.",
"func":1
},
{
"ref":"laueimproc.testing.tests.reciprocal",
"url":45,
"doc":"Test the convertion between primitive and reciprocal."
},
{
"ref":"laueimproc.testing.tests.reciprocal.test_batch_primitive_to_reciprocal",
"url":45,
"doc":"Tests batch dimension.",
"func":1
},
{
"ref":"laueimproc.testing.tests.reciprocal.test_jac_primitive_to_reciprocal",
"url":45,
"doc":"Tests compute jacobian.",
"func":1
},
{
"ref":"laueimproc.testing.tests.reciprocal.test_bij",
"url":45,
"doc":"Test is the transformation is reversible.",
"func":1
},
{
"ref":"laueimproc.testing.tests.rotation",
"url":46,
"doc":"Test the rotation function."
},
{
"ref":"laueimproc.testing.tests.rotation.test_bij_angle_to_rot_to_angle",
"url":46,
"doc":"Test if the function is bijective.",
"func":1
},
{
"ref":"laueimproc.testing.run",
"url":47,
"doc":"Executes all the tests via the  pytest module."
},
{
"ref":"laueimproc.testing.run.run_tests",
"url":47,
"doc":"Perform all unit tests.",
"func":1
},
{
"ref":"laueimproc.testing.coding_style",
"url":48,
"doc":"Execute tools to estimate the coding style quiality."
},
{
"ref":"laueimproc.testing.coding_style.test_mccabe_pycodestyle_pydocstyle_pyflakes",
"url":48,
"doc":"Run these linters throw pylama on laueimproc.",
"func":1
},
{
"ref":"laueimproc.testing.coding_style.test_pylint",
"url":48,
"doc":"Run pylint throw pylama on laueimproc.",
"func":1
},
{
"ref":"laueimproc.nn",
"url":49,
"doc":"Neuronal Network sub module."
},
{
"ref":"laueimproc.nn.dataaug",
"url":50,
"doc":"Image data augmentation."
},
{
"ref":"laueimproc.nn.dataaug.scale",
"url":51,
"doc":"Resize and image keeping the proportions."
},
{
"ref":"laueimproc.nn.dataaug.scale.rescale",
"url":51,
"doc":"Reshape the image, keep the spact ratio and pad with black pixels. Parameters      image : cutcutcodec.core.classes.image_video.FrameVideo or torch.Tensor or numpy.ndarray The image to be resized, of shape (height, width). shape : int and int The pixel dimensions of the returned image. The convention adopted is the numpy convention (height, width). copy : boolean, default=True If True, ensure that the returned tensor doesn't share the data of the input tensor. Returns    - resized_image The resized (and padded) image homogeneous with the input. The underground data are not shared with the input. A safe copy is done. Examples     >>> import torch >>> from laueimproc.nn.dataaug.scale import rescale >>> ref = torch.full 4, 8), 128, dtype=torch.uint8) >>> >>>  upscale >>> rescale(ref, (8, 12 tensor( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128], [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , dtype=torch.uint8) >>> >>>  downscale >>> rescale(ref, (4, 4 tensor( 0, 0, 0, 0], [128, 128, 128, 128], [128, 128, 128, 128], [ 0, 0, 0, 0 , dtype=torch.uint8) >>> >>>  mix >>> rescale(ref, (6, 6 tensor( 0, 0, 0, 0, 0, 0], [128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128], [ 0, 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0, 0 , dtype=torch.uint8) >>>",
"func":1
},
{
"ref":"laueimproc.nn.dataaug.patch",
"url":52,
"doc":"Crop and pad an image to schange the size without any interpolation."
},
{
"ref":"laueimproc.nn.dataaug.patch.patch",
"url":52,
"doc":"Pad the image with transparent borders. Parameters      image : torch.Tensor or numpy.ndarray The image to be cropped and padded. shape : int and int The pixel dimensions of the returned image. The convention adopted is the numpy convention (height, width). copy : boolean, default=True If True, ensure that the returned tensor doesn't share the data of the input tensor. Returns    - patched_image The cropped and padded image homogeneous with the input. Examples     >>> import torch >>> from laueimproc.nn.dataaug.patch import patch >>> ref = torch.full 4, 8), 128, dtype=torch.uint8) >>> patch(ref, (6, 6 tensor( 0, 0, 0, 0, 0, 0], [128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128], [128, 128, 128, 128, 128, 128], [ 0, 0, 0, 0, 0, 0 , dtype=torch.uint8) >>> patch(ref, (3, 9 tensor( 128, 128, 128, 128, 128, 128, 128, 128, 0], [128, 128, 128, 128, 128, 128, 128, 128, 0], [128, 128, 128, 128, 128, 128, 128, 128, 0 , dtype=torch.uint8) >>> patch(ref, (2, 2 tensor( 128, 128], [128, 128 , dtype=torch.uint8) >>> patch(ref, (10, 10 tensor( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 128, 128, 128, 128, 128, 128, 128, 128, 0], [ 0, 128, 128, 128, 128, 128, 128, 128, 128, 0], [ 0, 128, 128, 128, 128, 128, 128, 128, 128, 0], [ 0, 128, 128, 128, 128, 128, 128, 128, 128, 0], [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , dtype=torch.uint8) >>>",
"func":1
},
{
"ref":"laueimproc.nn.train",
"url":53,
"doc":"Training pipeline for the models."
},
{
"ref":"laueimproc.nn.train.train_vae_spot_classifier",
"url":53,
"doc":"Train the model. Parameters      model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier The initialised but not already trained model. dataset : laueimproc.classes.dataset.DiagramsDataset Contains all the initialised diagrams. batch : int, optional The number of pictures in each batch. By default, the batch size is equal to the dataset size, so that there is exactely one complete batch per epoch. epoch : int, default=10 The number of epoch. lr : float, optional The learning rate. fig : matplotlib.figure.Figure, optional An empty figure ready to be filled.",
"func":1
},
{
"ref":"laueimproc.nn.loader",
"url":54,
"doc":"Load and batch the data for neuronal network training."
},
{
"ref":"laueimproc.nn.loader.SpotDataloader",
"url":54,
"doc":"Get spots picture, apply dataaug and set in batch. Attributes      batch_size : int The batch dimension, read and write. Initialise the data loader. Parameters      dataset : laueimproc.classes.dataset.DiagramsDataset Contains all the initialised diagrams. model : laueimproc.nn.vae_spot_classifier.VAESpotClassifier The model, for the dataaug."
},
{
"ref":"laueimproc.nn.loader.SpotDataloader.batch_size",
"url":54,
"doc":"Return the batch size."
},
{
"ref":"laueimproc.nn.loader.find_shape",
"url":54,
"doc":"Scan the shape of all the spots and deduce the best shape. Parameters      dataset : laueimproc.classes.dataset.DiagramsDataset Contains all the initialised diagrams. percent : float The percentage of spots smaller or equal to the shape returned. Returns    - height : int The height shape. width : int The width shape. Examples     >>> import laueimproc >>> from laueimproc.nn.loader import find_shape >>> def init(diagram: laueimproc.Diagram):  . diagram.find_spots()  . >>> dataset = laueimproc.DiagramsDataset(laueimproc.io.get_samples( >>> _ = dataset.apply(init) >>> find_shape(dataset, 0.95) (15, 12) >>> find_shape(dataset, 0.5) (5, 5) >>>",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier",
"url":55,
"doc":"Classifier of laue spots using variational convolutive auto-encoder."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier",
"url":55,
"doc":"A partialy convolutive variationel auto encoder used for unsupervised spot classification. Attributes      decoder : Decoder The decoder part, able to random draw and reconstitue an image from the encoder. device: torch.device The device of the model. encoder : Encoder The encoder part, able to transform an image into au gaussian law. latent_dim : int The dimension of the latent space. shape : tuple[int, int] The shape of the rois. space : float, default = 3.0 The non penalized spreading area half size. Initialise the model. Parameters      shape : tuple[int, int] Transmitted to  laueimproc.nn.vae_spot_classifier.Encoder and  laueimproc.nn.vae_spot_classifier.Decoder . latent_dim : int, default=2 Transmitted to  laueimproc.nn.vae_spot_classifier.Encoder and  laueimproc.nn.vae_spot_classifier.Decoder . space : float, default=3.0 The non penalized spreading area in the latent space. All the points with abs(p) <= space are autorized. A small value condensate all the data, very continuous space but hard to split. In a other way, a large space split the clusters but the values betwean the clusters are not well defined. intensity_sensitive : boolean, default=True If set to False, the model will not consider the spots intensity, as they will be normalized to have a power of 1. scale_sensitive : boolean = True If set to False, the model will not consider the spots size, as they will be resized and reinterpolated to a constant shape."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.dataaug",
"url":55,
"doc":"Apply all the data augmentations on the image. Parameters      image : torch.Tensor The image of shape (h, w). Returns    - aug_batch : torch.Tensor The augmented stack of images of shape (n, 1, h', w').",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.device",
"url":55,
"doc":"Return the device of the model."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.forward",
"url":55,
"doc":"Encode, random draw and decode the image. Parameters      data : laueimproc.classes.base_diagram.BaseDiagram or torch.Tensor If a digram is provided, the spots are extract, data augmentation are applied, and the mean projection in the latent space is returned. If the input is a tensor, data augmentation are not applied. Return the autoencoded data, after having decoded a random latent space draw. Returns    - torch.Tensor The mean latent vector of shape (n, latent_dim) if the input is a Diagram. The generated image of shape (n, height, width) otherwise.",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.latent_dim",
"url":55,
"doc":"Return the dimension of the latent space."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.loss",
"url":55,
"doc":"Forward the data and compute the loss values. Parameters      batch : torch.Tensor The image stack of shape (n, h, w). Returns    - mse_loss : torch.Tensor The sum of the mean square error loss for each image in the batch, shape (1,). kld_loss : torch.Tensor Pretty close to the sum of the Kullback-Leibler divergence for each projection in the batch, shape (1,). It is not litteraly the Kullback-Leibler divergence because the peanality for the mean is less strict, the cost is 0 in the [-space, space] interval. The cost is minimum when var=1 and -space<=mean<=space. Notes   -  No verifications are performed for performance reason.  The reduction is sum and not mean because it ables to split the batch in several slices.",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.normalization",
"url":55,
"doc":"Return the mean and the std of all the training data."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.plot_autoencode",
"url":55,
"doc":"Encode and decode the images, plot the initial and regenerated images. Parameters      axe_input : matplotlib.axes.Axes The 2d empty axe ready to be filled by the input mosaic. axe_output : matplotlib.axes.Axes The 2d empty axe ready to be filled by the generated mosaic. spots : torch.Tensor The image stack of shape (n, h, w).",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.scan_data",
"url":55,
"doc":"Complete the data histogram to standardise data (centered and reduction). Parameters      spots_generator : iterable A generator of spot batch, each item has to be of shape (:, h, w).",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.shape",
"url":55,
"doc":"Return the shape of the images."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.VAESpotClassifier.space",
"url":55,
"doc":"Return the non penalized spreading area half size."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Decoder",
"url":55,
"doc":"Decode the latent sample into a new image. Attributes      parent : laueimproc.nn.vae_spot_classifier.VAESpotClassifier The main full auto encoder, containing this module. Initialise the decoder. Parameters      parent : laueimproc.nn.vae_spot_classifier.VAESpotClassifier The main module."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Decoder.forward",
"url":55,
"doc":"Generate a new image from the samples. Parameters      sample : torch.Tensor The batch of the n samples, output of the  Decoder.parametrize function. The size is (n, latent_dim) Returns    - image : torch.Tensor The generated image batch, of shape (n, 1, height, width) Notes   - No verifications are performed for performance reason.",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Decoder.parametrize",
"url":55,
"doc":"Perform a random draw according to the normal law N(mean, std 2). Parameters      mean : torch.Tensor The batch of the mean vectors, shape (n, latent_dim). std : torch.Tensor The batch of the diagonal sqrt(covariance) matrix, shape (n, latent_dim). Returns    - draw : torch.Tensor The batch of the random draw. Notes   - No verifications are performed for performance reason.",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Decoder.parent",
"url":55,
"doc":"Return the parent module."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Decoder.plot_map",
"url":55,
"doc":"Generate and display spots from a regular sampling of latent space. Parameters      axe : matplotlib.axes.Axes The 2d empty axe ready to be filled. grid : int or tuple[int, int] Grid dimension in latent space. If only one number is supplied, the grid will have this dimension on all axes. The 2 coordinates corresponds respectively to the number of lines and columns.",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Encoder",
"url":55,
"doc":"Encode an image into a gausian probality density. Attributes      parent : laueimproc.nn.vae_spot_classifier.VAESpotClassifier The main full auto encoder, containing this module. Initialise the encoder. Parameters      parent : laueimproc.nn.vae_spot_classifier.VAESpotClassifier The main module."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Encoder.forward",
"url":55,
"doc":"Extract the mean and the std for each images. Parameters      batch : torch.Tensor The stack of the n images, of shape (n, 1, height, width). Returns    - mean : torch.Tensor The mean (center of gaussians) for each image, shape (n, latent_dims). std : torch.Tensor The standard deviation (shape of gaussian) for each image, shape (n, latent_dims). Notes   - No verifications are performed for performance reason. If the model is in eval mode, it computes only the mean and gives the value None to the std.",
"func":1
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Encoder.parent",
"url":55,
"doc":"Return the parent module."
},
{
"ref":"laueimproc.nn.vae_spot_classifier.Encoder.plot_latent",
"url":55,
"doc":"Plot the 2d pca of the spots projected in the latent space. Parameters      axe : matplotlib.axes.Axes The 2d empty axe ready to be filled. spots_generator : iterable A generator of spot batch, each item has to be of shape (:, h, w).",
"func":1
},
{
"ref":"laueimproc.ml",
"url":56,
"doc":"General machine learning utils."
},
{
"ref":"laueimproc.ml.spot_dist",
"url":57,
"doc":"Find the close spots in two diagrams."
},
{
"ref":"laueimproc.ml.spot_dist.associate_spots",
"url":57,
"doc":"Find the close spots. Parameters      pos1 : torch.Tensor The coordinates of the position of each spot of the first diagram. Is is a tensor of shape (n, 2) containg real elements. pos2 : torch.Tensor The coordinates of the position of each spot of the second diagram. Is is a tensor of shape (n, 2) containg real elements. eps : float The max euclidian distance to associate 2 spots. Returns    - pair : torch.Tensor The couple of the indices of the closest spots, of shape (n, 2). Examples     >>> import torch >>> from laueimproc.ml.spot_dist import associate_spots >>> pos1 = torch.rand 1500, 2), dtype=torch.float32) >>> pos2 = torch.rand 2500, 2), dtype=torch.float32) >>> eps = 1e-3 >>> pair = associate_spots(pos1, pos2, eps) >>> (torch.sqrt pos1[pair[:, 0 - pos2[pair[:, 1 ) 2)  >>",
"func":1
},
{
"ref":"laueimproc.ml.spot_dist.spotslabel_to_diag",
"url":57,
"doc":"Inverse the representation, from diagram to spots. Parameters      labels : dict[int, torch.Tensor] To each diagram index, associate the label of each spot as a compact dict. Each value is a tensor of shape (n, 2), first column is native spot index into the diagram, then second column corresponds to the label. Returns    - diagrams : dict[int, set[int To each spot label, associate the set of diagram indices, containg the spot. Examples     >>> import torch >>> from laueimproc.ml.spot_dist import (associate_spots, track_spots,  . spotslabel_to_diag) >>> h, w = 5, 10 >>> diags = torch.tensor(  . sum  j, j+1] for j in range(i w, (i+1) w-1)] for i in range(h , start=[])  . + sum  j, j+w] for j in range(i w, (i+1) w)] for i in range(h-1 , start=[])  . ) >>> pos = [torch.rand 2000, 2), dtype=torch.float32) for _ in range(diags.max()+1)] >>> pairs = [associate_spots(pos[i], pos[j], 5e-3) for i, j in diags.tolist()] >>> labels = track_spots(pairs, diags) >>> diagrams = spotslabel_to_diag(labels) >>>",
"func":1
},
{
"ref":"laueimproc.ml.spot_dist.track_spots",
"url":57,
"doc":"Associate one label by position, give this label to all spots at this position. Parameters      pairs : list[torch.Tensor] For each pair of diagrams, contains the couple of close spots indices. diags : torch.Tensor The index of the diagrams for each pair, shape (len(pairs), 2). Returns    - labels : dict[int, torch.Tensor] To each diagram index, associate the label of each spot as a compact dict. Each value is a tensor of shape (n, 2), first column is native spot index into the diagram, then second column corresponds to the label. Examples     >>> import torch >>> from laueimproc.ml.spot_dist import associate_spots, track_spots >>> h, w = 5, 10 >>> diags = torch.tensor(  . sum  j, j+1] for j in range(i w, (i+1) w-1)] for i in range(h , start=[])  . + sum  j, j+w] for j in range(i w, (i+1) w)] for i in range(h-1 , start=[])  . ) >>> pos = [torch.rand 2000, 2), dtype=torch.float32) for _ in range(diags.max()+1)] >>> pairs = [associate_spots(pos[i], pos[j], 5e-3) for i, j in diags.tolist()] >>> labels = track_spots(pairs, diags) >>>",
"func":1
},
{
"ref":"laueimproc.ml.dataset_dist",
"url":58,
"doc":"Manage the diagams positioning inside a dataset."
},
{
"ref":"laueimproc.ml.dataset_dist.call_diag2scalars",
"url":58,
"doc":"Call the function, check, cast and return the output. The function typing is assumed to be checked. Parameters      pos_func : callable The function that associate a space position to a diagram index. index : int The argument value of the function. Returns    - position : tuple[float,  .] The scalar vector as a tuple of float.",
"func":1
},
{
"ref":"laueimproc.ml.dataset_dist.check_diag2scalars_typing",
"url":58,
"doc":"Ensure that the position function has the right type of input / outputs. Parameters      pos_func : callable A function supposed to take a diagram index as input and that return a scalar vector in a space of n dimensions. Raises    AssertionError If something wrong is detected. Examples     >>> import pytest >>> from laueimproc.ml.dataset_dist import check_diag2scalars_typing >>> def ok_1(index: int) -> float:  . return float(index)  . >>> def ok_2(index: int) -> tuple[float]:  . return (float(index),)  . >>> def ok_3(index: int) -> tuple[float, float]:  . return (float(index), 0.0)  . >>> def warn_1(index):  . return float(index)  . >>> def warn_2(index: int):  . return float(index)  . >>> def warn_3(index) -> float:  . return float(index)  . >>> def warn_4(index: int) -> tuple:  . return float(index)  . >>> error_1 = \"this is not a function\" >>> def error_2(file: str) -> float:  bad input type  . return float(file)  . >>> def error_3(index: int) -> list:  bad output type  . return [float(index)]  . >>> def error_4(index: int, cam: str) -> tuple:  bag input arguments  . return float(index)  . >>> check_diag2scalars_typing(ok_1) >>> check_diag2scalars_typing(ok_2) >>> check_diag2scalars_typing(ok_3) >>> >>> with pytest.warns(SyntaxWarning):  . check_diag2scalars_typing(warn_1)  . >>> with pytest.warns(SyntaxWarning):  . check_diag2scalars_typing(warn_2)  . >>> with pytest.warns(SyntaxWarning):  . check_diag2scalars_typing(warn_3)  . >>> with pytest.warns(SyntaxWarning):  . check_diag2scalars_typing(warn_4)  . >>> >>> with pytest.raises(AssertionError):  . check_diag2scalars_typing(error_1)  . >>> with pytest.raises(AssertionError):  . check_diag2scalars_typing(error_2)  . >>> with pytest.raises(AssertionError):  . check_diag2scalars_typing(error_3)  . >>> with pytest.raises(AssertionError):  . check_diag2scalars_typing(error_4)  . >>>",
"func":1
},
{
"ref":"laueimproc.ml.dataset_dist.select_closest",
"url":58,
"doc":"Select the closest point. Find the index i such as \\(d_i\\) is minimum, using the following formalism: \\(\\begin{cases} d_i = \\sqrt{\\sum\\limits_{j=0}^{D-1}\\left(\\kappa_j(p_j-x_{ij} ^2\\right)}  \\left|p_j-x_{ij}\\right| \\le \\epsilon_j, \\forall j \\in [\\![0;D-1]\\!]  \\end{cases}\\)  \\(D\\), the number of dimensions of the space used.  \\(\\kappa_j\\), a scalar inversely homogeneous has the unit used by the quantity of index \\(j\\).  \\(p_j\\), the coordinate \\(j\\) of the point of reference.  \\(x_{ij}\\), the \\(i\\)-th point of comparaison, coordinate \\(j\\). Parameters      coords : torch.Tensor The float32 points of each individual \\(\\text{coords[i, j]} = x_{ij}\\), of shape (n, \\(D\\ . point : tuple[float,  .] The point of reference in the destination space \\(point[j] = p_j\\). tol : tuple[float,  .], default inf The absolute tolerence value for each component (kind of manhattan distance). Such as \\(\\text{tol[j]} = \\epsilon_j\\). scale : tuple[float,  .], optional \\(\\text{scale[j]} = \\kappa_j\\), used for rescale each axis before to compute the euclidian distance. By default \\(\\kappa_j = 1, \\forall j \\in [\\![0;D-1]\\!]\\). Returns    - index: int The index \\(i\\) of the closest item \\(\\underset{i}{\\operatorname{argmin \\left(d\\right)\\). Raises    LookupError If no points match the criteria. Examples     >>> import torch >>> from laueimproc.ml.dataset_dist import select_closest >>> coords = torch.empty 1000, 3), dtype=torch.float32) >>> coords[:, 0] = torch.linspace(-1, 1, 1000) >>> coords[:, 1] = torch.linspace(-10, 10, 1000) >>> coords[:, 2] = torch.arange(1000) % 2 >>> select_closest(coords, (0.0, 0.0, 0.1 500 >>> select_closest(coords, (0.0, 0.0, 0.9 499 >>> select_closest(coords, (0.5, 5.0, 0.1 750 >>> select_closest(coords, (0.5, 5.0, 0.1), scale=(10, 1, 0.01 749 >>> select_closest(coords, (0.0, 0.0, 0.1), tol=(4/1000, 40/1000, 0.2 500 >>> try:  . select_closest(coords, (0.0, 0.0, 0.1), tol=(1/1000, 10/1000, 0.05  . except LookupError as err:  . print(err)  . no point match >>>",
"func":1
},
{
"ref":"laueimproc.ml.dataset_dist.select_closests",
"url":58,
"doc":"Select the closest points. Find all the indices i such as: \\(\\left|p_j-x_{ij}\\right| \\le \\epsilon_j, \\forall j \\in [\\![0;D-1]\\!]\\) Sorted the results byt increasing \\(d_i\\) such as: \\(d_i = \\sqrt{\\sum\\limits_{j=0}^{D-1}\\left(\\kappa_j(p_j-x_{ij} ^2\\right)}\\)  \\(D\\), the number of dimensions of the space used.  \\(\\kappa_j\\), a scalar inversely homogeneous has the unit used by the quantity of index \\(j\\).  \\(p_j\\), the coordinate \\(j\\) of the point of reference.  \\(x_{ij}\\), the \\(i\\)-th point of comparaison, coordinate \\(j\\). Parameters      coords : torch.Tensor The float32 points of each individual \\(\\text{coords[i, j]} = x_{ij}\\), of shape (n, \\(D\\ . point : tuple[float,  .], optional If provided, the point \\(point[j] = p_j\\) is used to calculate the distance and sort the results. By default, the point taken is equal to the average of the tol. tol : tuple[float | tuple[float, float],  .], default inf The absolute tolerence value for each component (kind of manhattan distance). Such as \\(\\text{tol[j][0]} = \\epsilon_{j-min}, \\text{tol[j][1]} = \\epsilon_{j-max}\\). scale : tuple[float,  .], optional \\(\\text{scale[j]} = \\kappa_j\\), used for rescale each axis before to compute the euclidian distance. By default \\(\\kappa_j = 1, \\forall j \\in [\\![0;D-1]\\!]\\). Returns    - indices : torch.Tensor The int32 list of the sorted coords indices.",
"func":1
},
{
"ref":"laueimproc.immix",
"url":59,
"doc":"Mixture of images."
},
{
"ref":"laueimproc.immix.mean",
"url":60,
"doc":"Compute the mean image of a batch of images."
},
{
"ref":"laueimproc.immix.mean.mean_stack",
"url":60,
"doc":"Compute the average image. Parameters      dataset : laueimproc.classes.dataset.DiagramsDataset The dataset containing all images. Returns    - torch.Tensor The average image of all the images contained in this dataset.",
"func":1
},
{
"ref":"laueimproc.immix.inter",
"url":61,
"doc":"Compute an intermediate image of a batch of images."
},
{
"ref":"laueimproc.immix.inter.MomentumMixer",
"url":61,
"doc":"Compute an average momenttume betweem the two closest candidates. Attributes      high : int The index max of the sorted stack. low : int The index min of the sorted stack. Precompute the momentum. Parameters      nbr : int The number of items in the stack. level : float The relative position in the stack in [0, 1]. Examples     >>> import torch >>> from laueimproc.immix.inter import MomentumMixer >>> mom = MomentumMixer(5, 0.5) >>> mom(torch.arange(5)[mom.low], torch.arange(5)[mom.high]) tensor(2.) >>> mom = MomentumMixer(4, 0.5) >>> mom(torch.arange(4)[mom.low], torch.arange(4)[mom.high]) tensor(1.5000) >>> mom = MomentumMixer(101, 0.1 torch.pi) >>> mom(torch.arange(101)[mom.low], torch.arange(101)[mom.high]) tensor(31.4159) >>>"
},
{
"ref":"laueimproc.immix.inter.MomentumMixer.high",
"url":61,
"doc":"Return the index max of the sorted stack."
},
{
"ref":"laueimproc.immix.inter.MomentumMixer.low",
"url":61,
"doc":"Return the index min of the sorted stack."
},
{
"ref":"laueimproc.immix.inter.snowflake_stack",
"url":61,
"doc":"Compute the median, first quartile, third quartile or everything in between. This algorithm consists of computing the histogram of all the images into a heap of size n. Then compute the cumulative histogram to deduce in each slice the value is. To bound the result. Iterate the processus to refine the bounds until reaching the required accuracy. Parameters      dataset : laueimproc.classes.dataset.DiagramsDataset The dataset containing all images. level : float, default=0.5 The level of the sorted stack.  0 -> min filter  0.25 -> first quartile  0.5 (default) -> median  0.75 -> third quartile  1 -> max filter tol : float, default=1/(2 16-1) Accuracy of the estimated returned image. Returns    - torch.Tensor The 2d float32 grayscale image. Notes   - Unlike the native algorithm, images are read a number of times proportional to the logarithm of the inverse of the precision. Independent of the number of images in the dataset. This algorithm is therefore better suited to large datasets.",
"func":1
},
{
"ref":"laueimproc.immix.inter.sort_stack",
"url":61,
"doc":"Compute the median, first quartile, third quartile or everything in between. This algorithm consists of stacking all the images into a heap of size n. Then sort each column in the stack (as many columns as there are pixels in the image). Finally, we return the image in the new stack at height n   level . Parameters      dataset : laueimproc.classes.dataset.DiagramsDataset The dataset containing all images. level : float, default=0.5 The level of the sorted stack.  0 -> min filter  0.25 -> first quartile  0.5 (default) -> median  0.75 -> third quartile  1 -> max filter Returns    - torch.Tensor The 2d float32 grayscale image. Notes   - For reasons of memory limitations, the final image is calculated in small chunks. As a result, each image on the hard disk is read n times, with n proportional to the number of diagrams in the dataset.",
"func":1
},
{
"ref":"laueimproc.common",
"url":62,
"doc":"Little common tools."
},
{
"ref":"laueimproc.common.bytes2human",
"url":62,
"doc":"Convert a size in bytes in readable human string. Examples     >>> from laueimproc.common import bytes2human >>> bytes2human(0) '0.0B' >>> bytes2human(2000) '2.0kB' >>> bytes2human(2_000_000) '2.0MB' >>> bytes2human(2e9) '2.0GB' >>>",
"func":1
},
{
"ref":"laueimproc.common.get_project_root",
"url":62,
"doc":"Return the absolute project root folder. Examples     >>> from laueimproc.common import get_project_root >>> root = get_project_root() >>> root.is_dir() True >>> root.name 'laueimproc' >>>",
"func":1
},
{
"ref":"laueimproc.common.time2sec",
"url":62,
"doc":"Parse a time duration expression and return it in seconds. Raises    ValueError If the provided time dosen't match a parsable correct time format. Examples     >>> from laueimproc.common import time2sec >>> time2sec(12.34) 12.34 >>> time2sec(\"12.34\") 12.34 >>> time2sec(\".34\") 0.34 >>> time2sec(\"12.\") 12.0 >>> time2sec(\"12\") 12.0 >>> time2sec(\"12.34s\") 12.34 >>> time2sec(\"12.34 sec\") 12.34 >>> time2sec(\"2 m\") 120.0 >>> time2sec(\"2min 2\") 122.0 >>> time2sec(\" 2.5 h \") 9000.0 >>> time2sec(\"2hour02\") 7320.0 >>> time2sec(\"2h 2s\") 7202.0 >>> time2sec(\"2.5 hours 2.0 minutes 12.34 seconds\") 9132.34 >>>",
"func":1
},
{
"ref":"laueimproc.geometry",
"url":63,
"doc":"Implement the Bragg diffraction rules. Bases   -  \\(\\mathcal{B^c}\\): The orthonormal base of the crystal \\([\\mathbf{C_1}, \\mathbf{C_2}, \\mathbf{C_3}]\\).  \\(\\mathcal{B^l}\\): The orthonormal base of the lab \\([\\mathbf{L_1}, \\mathbf{L_2}, \\mathbf{L_3}]\\) in pyfai. Lattice parameters           \\([a, b, c, \\alpha, \\beta, \\gamma]\\): The lattice scalars parameters.  \\(\\mathbf{A}\\): The primitive column vectors \\([\\mathbf{e_1}, \\mathbf{e_2}, \\mathbf{e_3}]\\) in an orthonormal base.  \\(\\mathbf{B}\\): The reciprocal column vectors \\([\\mathbf{e_1^ }, \\mathbf{e_2^ }, \\mathbf{e_3^ }]\\) in an orthonormal base."
},
{
"ref":"laueimproc.geometry.hkl_reciprocal_to_energy",
"url":63,
"doc":"Alias to  laueimproc.geometry.bragg.hkl_reciprocal_to_uq_energy .",
"func":1
},
{
"ref":"laueimproc.geometry.hkl_reciprocal_to_uq",
"url":63,
"doc":"Alias to  laueimproc.geometry.bragg.hkl_reciprocal_to_uq_energy .",
"func":1
},
{
"ref":"laueimproc.geometry.hkl_reciprocal_to_uq_energy",
"url":63,
"doc":"Thanks to the bragg relation, compute the energy of each diffracted ray. Parameters      hkl : torch.Tensor The h, k, l indices of shape (\\ n, 3) we want to mesure. reciprocal : torch.Tensor Matrix \\(\\mathbf{B}\\) of shape (\\ r, 3, 3) in the lab base \\(\\mathcal{B^l}\\). cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape are (\\ n, \\ r, 3) and (\\ n, \\ r).  False: The final shape are (\\ broadcast(n, r), 3) and broadcast(n, r). Returns    - u_q : torch.Tensor All the unitary diffracting plane normal vector of shape ( ., 3). The vectors are expressed in the same base as the reciprocal space. energy : torch.Tensor The energy of each ray in J as a tensor of shape ( .). \\(\\begin{cases} E = \\frac{hc}{\\lambda}  \\lambda = 2d\\sin(\\theta)  \\sin(\\theta) = \\left| \\langle u_i, u_q \\rangle \\right|  \\end{cases}\\) Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\). Examples     >>> import torch >>> from laueimproc.geometry.bragg import hkl_reciprocal_to_uq_energy >>> reciprocal = torch.torch.tensor( 1.6667e+09, 0.0000e+00, 0.0000e+00],  . [ 9.6225e+08, 3.0387e+09, -0.0000e+00],  . [-6.8044e+08, -2.1488e+09, 8.1653e+08 ) >>> hkl = torch.tensor([1, 2, -1]) >>> u_q, energy = hkl_reciprocal_to_uq_energy(hkl, reciprocal) >>> u_q tensor([ 0.1798, 0.7595, -0.6252]) >>> 6.24e18  energy  convertion J -> eV tensor(9200.6816) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.uf_to_uq",
"url":63,
"doc":"Calculate the vector normal to the diffracting planes. Bijection of  laueimproc.geometry.bragg.uq_to_uf . \\(u_q \\propto u_f - u_i\\) Parameters      u_f : torch.Tensor The unitary diffracted rays of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Returns    - u_q : torch.Tensor The unitary normals of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).",
"func":1
},
{
"ref":"laueimproc.geometry.uq_to_uf",
"url":63,
"doc":"Calculate the diffracted ray from q vector. Bijection of  laueimproc.geometry.bragg.uf_to_uq . \\(\\begin{cases} u_f - u_i = \\eta u_q  \\eta = 2 \\langle u_q, -u_i \\rangle  \\end{cases}\\) Parameters      u_q : torch.Tensor The unitary q vectors of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Returns    - u_f : torch.Tensor The unitary diffracted ray of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Notes   -  \\(u_f\\) is not sensitive to the \\(u_q\\) orientation.  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).",
"func":1
},
{
"ref":"laueimproc.geometry.select_hkl",
"url":63,
"doc":"Reject the hkl sistematicaly out of energy band. Parameters      reciprocal : torch.Tensor, optional Matrix \\(\\mathbf{B}\\) in any orthonormal base. max_hkl : int, optional The maximum absolute hkl sum such as |h| + |k| + |l|  >> import torch >>> from laueimproc.geometry.hkl import select_hkl >>> reciprocal = torch.tensor( 2.7778e9, 0, 0],  . [1.2142e2, 2.7778e9, 0],  . [1.2142e2, 1.2142e2, 2.7778e9 ) >>> select_hkl(max_hkl=18) tensor( 0, -18, 0], [ 0, -17, -1], [ 0, -17, 0],  ., [ 17, 0, 1], [ 17, 1, 0], [ 18, 0, 0 , dtype=torch.int16) >>> len(_) 4578 >>> select_hkl(max_hkl=18, keep_harmonics=False) tensor( 0, -17, -1], [ 0, -17, 1], [ 0, -16, -1],  ., [ 17, 0, -1], [ 17, 0, 1], [ 17, 1, 0 , dtype=torch.int16) >>> len(_) 3661 >>> select_hkl(reciprocal, e_max=20e3  1.60e-19)  20 keV tensor( 0, -11, -3], [ 0, -11, -2], [ 0, -11, -1],  ., [ 11, 3, 0], [ 11, 3, 1], [ 11, 3, 2 , dtype=torch.int16) >>> len(_) 1463 >>> select_hkl(reciprocal, e_max=20e3  1.60e-19, keep_harmonics=False)  20 keV tensor( 0, -11, -1], [ 0, -11, 1], [ 0, -10, -1],  ., [ 11, 0, -1], [ 11, 0, 1], [ 11, 1, 0 , dtype=torch.int16) >>> len(_) 1149 >>>",
"func":1
},
{
"ref":"laueimproc.geometry.lattice_to_primitive",
"url":63,
"doc":"Convert the lattice parameters into primitive vectors.  image  / / /build/media/IMGLatticeBc.avif Parameters      lattice : torch.Tensor The array of lattice parameters of shape ( ., 6). Values are \\([a, b, c, \\alpha, \\beta, \\gamma \\) in meters and radians. Returns    - primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) of shape ( ., 3, 3) in the crystal base \\(\\mathcal{B^c}\\). Examples     >>> import torch >>> from laueimproc.geometry.lattice import lattice_to_primitive >>> lattice = torch.tensor([6.0e-10, 3.8e-10, 15e-10, torch.pi/3, torch.pi/2, 2 torch.pi/3]) >>> lattice_to_primitive(lattice) tensor( 6.0000e-10, -1.9000e-10, -6.5567e-17], [ 0.0000e+00, 3.2909e-10, 8.6603e-10], [ 0.0000e+00, 0.0000e+00, 1.2247e-09 ) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.primitive_to_lattice",
"url":63,
"doc":"Convert the primitive vectors to the lattice parameters. Bijection of  laueimproc.geometry.lattice.lattice_to_primitive .  image  / / /build/media/IMGLattice.avif Parameters      primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) in any orthonormal base. Returns    - lattice : torch.Tensor The array of lattice parameters of shape ( ., 6). Values are \\([a, b, c, \\alpha, \\beta, \\gamma]\\) in meters and radians. Notes   - We have always  primitive_to_lattice(lattice_to_primitive(lattice  lattice , but it is not always the case for the inverse composition because the numerical value of \\(\\mathbf{A}\\) is expressed in the crystal base \\(\\mathcal{B^c}\\). Examples     >>> import torch >>> from laueimproc.geometry.lattice import primitive_to_lattice >>> primitive = torch.tensor( 6.0000e-10, -1.9000e-10, -6.5567e-17],  . [ 0.0000e+00, 3.2909e-10, 8.6603e-10],  . [ 0.0000e+00, 0.0000e+00, 1.2247e-09 ) >>> primitive_to_lattice(primitive)  quartz lattice tensor([6.0000e-10, 3.8000e-10, 1.5000e-09, 1.0472e+00, 1.5708e+00, 2.0944e+00]) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.compute_matching_rate",
"url":63,
"doc":"Compute the matching rate. It is the number of ray in  theo_uq , close engouth to at least one ray of  exp_uq . Parameters      exp_uq : torch.Tensor The unitary experimental uq vector of shape (\\ n, e, 3). theo_uq : torch.Tensor The unitary simulated theorical uq vector of shape (\\ n', t, 3). phi_max : float The maximum positive angular distance in radian to consider that the rays are closed enough. Returns    - match : torch.Tensor The matching rate of shape broadcast(n, n'). Examples     >>> import torch >>> from laueimproc.geometry.metric import compute_matching_rate >>> exp_uq = torch.randn(1000, 3) >>> exp_uq /= torch.linalg.norm(exp_uq, dim=1, keepdims=True) >>> theo_uq = torch.randn(5000, 3) >>> theo_uq /= torch.linalg.norm(theo_uq, dim=1, keepdims=True) >>> rate = compute_matching_rate(exp_uq, theo_uq, 0.5  torch.pi/180) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.compute_matching_rate_continuous",
"url":63,
"doc":"Compute the matching rate. This is a continuity extension of the disctere function  laueimproc.geometry.metric.compute_matching_rate . Let \\(\\phi\\) be the angle between two rays. The matching rate is defined with \\(r = \\sum f(\\phi_i)\\) with \\(f(\\phi) = e^{-\\frac{\\log(2)}{\\phi_{max \\phi}\\). Parameters      exp_uq : torch.Tensor The unitary experimental uq vector of shape (\\ n, e, 3). theo_uq : torch.Tensor The unitary simulated theorical uq vector of shape (\\ n', t, 3). phi_max : float The maximum positive angular distance in radian to consider that the rays are closed enough. Returns    - rate : torch.Tensor The continuous matching rate of shape broadcast(n, n'). Examples     >>> import torch >>> from laueimproc.geometry.metric import compute_matching_rate_continuous >>> exp_uq = torch.randn(1000, 3) >>> exp_uq /= torch.linalg.norm(exp_uq, dim=1, keepdims=True) >>> theo_uq = torch.randn(5000, 3) >>> theo_uq /= torch.linalg.norm(theo_uq, dim=1, keepdims=True) >>> rate = compute_matching_rate_continuous(exp_uq, theo_uq, 0.5  torch.pi/180) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.detector_to_ray",
"url":63,
"doc":"Find light ray witch intersected on the detector. Bijection of  laueimproc.geometry.projection.ray_to_detector . Parameters      point : torch.Tensor The 2d point in meter in the referencial of the detector of shape (\\ r, \\ p, 2). poni : torch.Tensor The point of normal incidence, callibration parameters according the pyfai convention. Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\\ p', 6). Returns    - ray : torch.Tensor The unitary ray vector of shape (\\ r, \\ broadcast(p, p'), 3).",
"func":1
},
{
"ref":"laueimproc.geometry.ray_to_detector",
"url":63,
"doc":"Find the intersection of the light ray with the detector. Bijection of  laueimproc.geometry.projection.detector_to_ray . Parameters      ray : torch.Tensor The unitary ray vector of shape (\\ r, 3). poni : torch.Tensor The point of normal incidence, callibration parameters according the pyfai convention. Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\\ p, 6). cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape are (\\ r, \\ p, 2) and (\\ r, \\ p).  False: The final shape are (\\ broadcast(r, p), 2) and broadcast(r, p). Returns    - point : torch.Tensor The 2d points in meter in the referencial of the detector of shape ( ., 2). dist : torch.Tensor The algebrical distance of the ray between the sample and the detector. a positive value means that the beam crashs on the detector. A negative value means it is moving away. The shape is ( .).",
"func":1
},
{
"ref":"laueimproc.geometry.primitive_to_reciprocal",
"url":63,
"doc":"Convert the primitive vectors into the reciprocal base vectors. Bijection of  laueimproc.geometry.reciprocal.reciprocal_to_primitive .  image  / / /build/media/IMGPrimitiveReciprocal.avif Parameters      primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) in any orthonormal base. Returns    - reciprocal : torch.Tensor Matrix \\(\\mathbf{B}\\) in the same orthonormal base. Examples     >>> import torch >>> from laueimproc.geometry.reciprocal import primitive_to_reciprocal >>> primitive = torch.tensor( 6.0000e-10, -1.9000e-10, -6.5567e-17],  . [ 0.0000e+00, 3.2909e-10, 8.6603e-10],  . [ 0.0000e+00, 0.0000e+00, 1.2247e-09 ) >>> primitive_to_reciprocal(primitive) tensor( 1.6667e+09, 0.0000e+00, 0.0000e+00], [ 9.6225e+08, 3.0387e+09, -0.0000e+00], [-6.8044e+08, -2.1488e+09, 8.1653e+08 ) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.reciprocal_to_primitive",
"url":63,
"doc":"Convert the reciprocal vectors into the primitive base vectors. Bijection of  laueimproc.geometry.reciprocal.primitive_to_reciprocal .  image  / / /build/media/IMGPrimitiveReciprocal.avif Parameters      reciprocal : torch.Tensor Matrix \\(\\mathbf{B}\\) in any orthonormal base. Returns    - primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) in the same orthonormal base.",
"func":1
},
{
"ref":"laueimproc.geometry.angle_to_rot",
"url":63,
"doc":"Generate a rotation matrix from the given angles. The rotation are following the pyfai convention. Parameters      theta1 : torch.Tensor or float The first rotation angle of shape (\\ a,). \\( rot_1 = \\begin{pmatrix} 1 & 0 & 0  0 & \\cos(\\theta_1) & -\\sin(\\theta_1)  c & \\sin(\\theta_1) & \\cos(\\theta_1)  \\end{pmatrix} \\) theta2 : torch.Tensor or float The second rotation angle of shape (\\ b,). \\( rot_2 = \\begin{pmatrix} \\cos(\\theta_2) & 0 & \\sin(\\theta_2)  0 & 1 & 0  -\\sin(\\theta_2) & 0 & \\cos(\\theta_2)  \\end{pmatrix} \\) theta3 : torch.Tensor or float The third rotation angle of shape (\\ c,). (inverse of pyfai convention) \\( rot_3 = \\begin{pmatrix} \\cos(\\theta_3) & -\\sin(\\theta_3) & 0  \\sin(\\theta_3) & \\cos(\\theta_3) & 0  0 & 0 & 1  \\end{pmatrix} \\) cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape is (\\ a, \\ b, \\ c, 3, 3).  False: The final shape is (\\ broadcast(a, b, c), 3, 3). Returns    - rot : torch.Tensor The global rotation \\(rot_3 . rot_2 . rot_1\\). Examples     >>> import torch >>> from laueimproc.geometry.rotation import angle_to_rot >>> angle_to_rot(theta1=torch.pi/6, theta2=torch.pi/6, theta3=torch.pi/6) tensor( 0.7500, -0.2165, 0.6250], [ 0.4330, 0.8750, -0.2165], [-0.5000, 0.4330, 0.7500 ) >>> angle_to_rot(torch.randn(4), torch.randn(5, 6), torch.randn(7, 8, 9 .shape torch.Size([4, 5, 6, 7, 8, 9, 3, 3]) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.rot_to_angle",
"url":63,
"doc":"Extract the rotation angles from a fulle rotation matrix. Bijection of  laueimproc.geometry.rotation.angle_to_rot . Parameters      rot : torch.Tensor The rotation matrix \\(rot_3 . rot_2 . rot_1\\) of shape ( ., 3, 3). Returns    - theta1 : torch.Tensor or float The first rotation angle of shape ( .). \\(\\theta_1 \\in [-\\pi, \\pi]\\) \\( rot_1 = \\begin{pmatrix} 1 & 0 & 0  0 & \\cos(\\theta_1) & -\\sin(\\theta_1)  c & \\sin(\\theta_1) & \\cos(\\theta_1)  \\end{pmatrix} \\) theta2 : torch.Tensor or float The second rotation angle of shape ( .). \\(\\theta_2 \\in [-\\frac{\\pi}{2}, \\frac{\\pi}{2}]\\) \\( rot_2 = \\begin{pmatrix} \\cos(\\theta_2) & 0 & \\sin(\\theta_2)  0 & 1 & 0  -\\sin(\\theta_2) & 0 & \\cos(\\theta_2)  \\end{pmatrix} \\) theta3 : torch.Tensor or float The third rotation angle of shape ( .). (inverse of pyfai convention) \\(\\theta_3 \\in [-\\pi, \\pi]\\) \\( rot_3 = \\begin{pmatrix} \\cos(\\theta_3) & -\\sin(\\theta_3) & 0  \\sin(\\theta_3) & \\cos(\\theta_3) & 0  0 & 0 & 1  \\end{pmatrix} \\) Examples     >>> import torch >>> from laueimproc.geometry.rotation import rot_to_angle >>> rot = torch.tensor( 0.7500, -0.2165, 0.6250],  . [ 0.4330, 0.8750, -0.2165],  . [-0.5000, 0.4330, 0.7500 ) >>> theta1, theta2, theta3 = rot_to_angle(rot) >>> torch.rad2deg(theta1).round() tensor(30.) >>> torch.rad2deg(theta2).round() tensor(30.) >>> torch.rad2deg(theta3).round() tensor(30.) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.rotate_crystal",
"url":63,
"doc":"Apply an active rotation to the crystal. Parameters      crystal : torch.Tensor The primitive \\ \\mathbf{A})\\) or reciprocal \\ \\mathbf{B})\\) in the base \\(\\mathcal{B}\\). The shape of this parameter is (\\ c, 3, 3). rot : torch.Tensor The active rotation matrix, of shape (\\ r, 3, 3). cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape is (\\ c, \\ r, 3, 3).  False: The final shape is (\\ broadcast(c, r), 3, 3). Returns    - rotated_crystal: torch.Tensor The batched matricial product  rot @ crystal .",
"func":1
},
{
"ref":"laueimproc.geometry.thetachi_to_uf",
"url":63,
"doc":"Reconstruct the diffracted ray from the deviation angles. Bijection of  laueimproc.geometry.thetachi.uf_to_thetachi .  image  / / /build/media/IMGThetaChi.avif Parameters      theta : torch.Tensor The half deviation angle in radian, with \\(\\theta \\in [0, \\frac{\\pi}{2}]\\). chi : torch.Tensor The rotation angle in radian from the vertical plan \\ \\mathbf{L_1}, \\mathbf{L_3})\\) to the plan \\ u_i, \\mathcal{B^l_z})\\), with \\(\\chi \\in [-\\pi, \\pi]\\). Returns    - u_f : torch.Tensor The unitary diffracted ray of shape ( broadcast(theta.shape, chi.shape), 3). It is expressed in the lab base \\(\\mathcal{B^l}\\). Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).  This function is slow, use  laueimproc.geometry.bragg.uq_to_uf if you can. Examples     >>> import torch >>> from laueimproc.geometry.thetachi import thetachi_to_uf >>> theta = torch.deg2rad(torch.tensor([15., 30., 45.] >>> chi = torch.deg2rad(torch.tensor([ -0., 90., -45.] >>> thetachi_to_uf(theta, chi).round(decimals=4) tensor( 0.5000, 0.0000, 0.8660], [-0.0000, -0.8660, 0.5000], [ 0.7071, 0.7071, -0.0000 ) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.uf_to_thetachi",
"url":63,
"doc":"Find the angular deviation of the dffracted ray. Bijection of  laueimproc.geometry.thetachi.thetachi_to_uf .  image  / / /build/media/IMGThetaChi.avif Parameters      u_f : torch.Tensor The unitary diffracted ray of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Returns    - theta : torch.Tensor The half deviation angle in radian, of shape ( .). \\(\\theta \\in [0, \\frac{\\pi}{2}]\\) chi : torch.Tensor The counterclockwise (trigonometric) rotation of the diffracted ray if you look as u_i. It is the angle from the vertical plan \\ \\mathbf{L_1}, \\mathbf{L_3})\\) to the plan \\ u_i, \\mathcal{B^l_z})\\). The shape is ( .) as well. \\(\\chi \\in [-\\pi, \\pi]\\) Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).  This function is slow, use  laueimproc.geometry.bragg.uf_to_uq if you can. Examples     >>> import torch >>> from laueimproc.geometry.thetachi import uf_to_thetachi >>> u_f = torch.tensor( 1/2, 0, 3 (1/2)/2], [0, -3 (1/2)/2, 1/2], [2 (1/2), 2 (1/2), 0 ) >>> theta, chi = uf_to_thetachi(u_f) >>> torch.rad2deg(theta).round() tensor([15., 30., 45.]) >>> torch.rad2deg(chi).round() tensor([ -0., 90., -45.]) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.metric",
"url":64,
"doc":"Implement some loss functions."
},
{
"ref":"laueimproc.geometry.metric.raydotdist",
"url":64,
"doc":"Compute the scalar product matrix of the rays pairwise. Parameters      ray_point_1 : torch.Tensor The 2d point associated to uf in the referencial of the detector of shape (\\ n, r1, 2). Could be directly the unitary ray vector uf or uq of shape (\\ n, r1, 3). ray_point_2 : torch.Tensor The 2d point associated to uf in the referencial of the detector of shape (\\ n', r2, 2 . Could be directly the unitary ray vector uf or uq of shape (\\ n', r2, 3). poni : torch.Tensor, optional Only use if the ray are projected points. The point of normal incidence, callibration parameters according the pyfai convention. Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\\ p, 6). Returns    - dist : torch.Tensor The distance matrix \\(\\cos(\\phi)\\) of shape (\\ broadcast(n, n'), \\ p, r1, r2).",
"func":1
},
{
"ref":"laueimproc.geometry.metric.compute_matching_rate",
"url":64,
"doc":"Compute the matching rate. It is the number of ray in  theo_uq , close engouth to at least one ray of  exp_uq . Parameters      exp_uq : torch.Tensor The unitary experimental uq vector of shape (\\ n, e, 3). theo_uq : torch.Tensor The unitary simulated theorical uq vector of shape (\\ n', t, 3). phi_max : float The maximum positive angular distance in radian to consider that the rays are closed enough. Returns    - match : torch.Tensor The matching rate of shape broadcast(n, n'). Examples     >>> import torch >>> from laueimproc.geometry.metric import compute_matching_rate >>> exp_uq = torch.randn(1000, 3) >>> exp_uq /= torch.linalg.norm(exp_uq, dim=1, keepdims=True) >>> theo_uq = torch.randn(5000, 3) >>> theo_uq /= torch.linalg.norm(theo_uq, dim=1, keepdims=True) >>> rate = compute_matching_rate(exp_uq, theo_uq, 0.5  torch.pi/180) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.metric.compute_matching_rate_continuous",
"url":64,
"doc":"Compute the matching rate. This is a continuity extension of the disctere function  laueimproc.geometry.metric.compute_matching_rate . Let \\(\\phi\\) be the angle between two rays. The matching rate is defined with \\(r = \\sum f(\\phi_i)\\) with \\(f(\\phi) = e^{-\\frac{\\log(2)}{\\phi_{max \\phi}\\). Parameters      exp_uq : torch.Tensor The unitary experimental uq vector of shape (\\ n, e, 3). theo_uq : torch.Tensor The unitary simulated theorical uq vector of shape (\\ n', t, 3). phi_max : float The maximum positive angular distance in radian to consider that the rays are closed enough. Returns    - rate : torch.Tensor The continuous matching rate of shape broadcast(n, n'). Examples     >>> import torch >>> from laueimproc.geometry.metric import compute_matching_rate_continuous >>> exp_uq = torch.randn(1000, 3) >>> exp_uq /= torch.linalg.norm(exp_uq, dim=1, keepdims=True) >>> theo_uq = torch.randn(5000, 3) >>> theo_uq /= torch.linalg.norm(theo_uq, dim=1, keepdims=True) >>> rate = compute_matching_rate_continuous(exp_uq, theo_uq, 0.5  torch.pi/180) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.bragg",
"url":65,
"doc":"Simulation of the bragg diffraction."
},
{
"ref":"laueimproc.geometry.bragg.hkl_reciprocal_to_energy",
"url":65,
"doc":"Alias to  laueimproc.geometry.bragg.hkl_reciprocal_to_uq_energy .",
"func":1
},
{
"ref":"laueimproc.geometry.bragg.hkl_reciprocal_to_uq",
"url":65,
"doc":"Alias to  laueimproc.geometry.bragg.hkl_reciprocal_to_uq_energy .",
"func":1
},
{
"ref":"laueimproc.geometry.bragg.hkl_reciprocal_to_uq_energy",
"url":65,
"doc":"Thanks to the bragg relation, compute the energy of each diffracted ray. Parameters      hkl : torch.Tensor The h, k, l indices of shape (\\ n, 3) we want to mesure. reciprocal : torch.Tensor Matrix \\(\\mathbf{B}\\) of shape (\\ r, 3, 3) in the lab base \\(\\mathcal{B^l}\\). cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape are (\\ n, \\ r, 3) and (\\ n, \\ r).  False: The final shape are (\\ broadcast(n, r), 3) and broadcast(n, r). Returns    - u_q : torch.Tensor All the unitary diffracting plane normal vector of shape ( ., 3). The vectors are expressed in the same base as the reciprocal space. energy : torch.Tensor The energy of each ray in J as a tensor of shape ( .). \\(\\begin{cases} E = \\frac{hc}{\\lambda}  \\lambda = 2d\\sin(\\theta)  \\sin(\\theta) = \\left| \\langle u_i, u_q \\rangle \\right|  \\end{cases}\\) Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\). Examples     >>> import torch >>> from laueimproc.geometry.bragg import hkl_reciprocal_to_uq_energy >>> reciprocal = torch.torch.tensor( 1.6667e+09, 0.0000e+00, 0.0000e+00],  . [ 9.6225e+08, 3.0387e+09, -0.0000e+00],  . [-6.8044e+08, -2.1488e+09, 8.1653e+08 ) >>> hkl = torch.tensor([1, 2, -1]) >>> u_q, energy = hkl_reciprocal_to_uq_energy(hkl, reciprocal) >>> u_q tensor([ 0.1798, 0.7595, -0.6252]) >>> 6.24e18  energy  convertion J -> eV tensor(9200.6816) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.bragg.uf_to_uq",
"url":65,
"doc":"Calculate the vector normal to the diffracting planes. Bijection of  laueimproc.geometry.bragg.uq_to_uf . \\(u_q \\propto u_f - u_i\\) Parameters      u_f : torch.Tensor The unitary diffracted rays of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Returns    - u_q : torch.Tensor The unitary normals of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).",
"func":1
},
{
"ref":"laueimproc.geometry.bragg.uq_to_uf",
"url":65,
"doc":"Calculate the diffracted ray from q vector. Bijection of  laueimproc.geometry.bragg.uf_to_uq . \\(\\begin{cases} u_f - u_i = \\eta u_q  \\eta = 2 \\langle u_q, -u_i \\rangle  \\end{cases}\\) Parameters      u_q : torch.Tensor The unitary q vectors of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Returns    - u_f : torch.Tensor The unitary diffracted ray of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Notes   -  \\(u_f\\) is not sensitive to the \\(u_q\\) orientation.  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).",
"func":1
},
{
"ref":"laueimproc.geometry.reciprocal",
"url":66,
"doc":"Enables communication between primitive \\(\\mathbf{A}\\) and reciprocal space \\(\\mathbf{B}\\)."
},
{
"ref":"laueimproc.geometry.reciprocal.primitive_to_reciprocal",
"url":66,
"doc":"Convert the primitive vectors into the reciprocal base vectors. Bijection of  laueimproc.geometry.reciprocal.reciprocal_to_primitive .  image  / / /build/media/IMGPrimitiveReciprocal.avif Parameters      primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) in any orthonormal base. Returns    - reciprocal : torch.Tensor Matrix \\(\\mathbf{B}\\) in the same orthonormal base. Examples     >>> import torch >>> from laueimproc.geometry.reciprocal import primitive_to_reciprocal >>> primitive = torch.tensor( 6.0000e-10, -1.9000e-10, -6.5567e-17],  . [ 0.0000e+00, 3.2909e-10, 8.6603e-10],  . [ 0.0000e+00, 0.0000e+00, 1.2247e-09 ) >>> primitive_to_reciprocal(primitive) tensor( 1.6667e+09, 0.0000e+00, 0.0000e+00], [ 9.6225e+08, 3.0387e+09, -0.0000e+00], [-6.8044e+08, -2.1488e+09, 8.1653e+08 ) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.reciprocal.reciprocal_to_primitive",
"url":66,
"doc":"Convert the reciprocal vectors into the primitive base vectors. Bijection of  laueimproc.geometry.reciprocal.primitive_to_reciprocal .  image  / / /build/media/IMGPrimitiveReciprocal.avif Parameters      reciprocal : torch.Tensor Matrix \\(\\mathbf{B}\\) in any orthonormal base. Returns    - primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) in the same orthonormal base.",
"func":1
},
{
"ref":"laueimproc.geometry.projection",
"url":67,
"doc":"Project rays on a physical or virtual plane."
},
{
"ref":"laueimproc.geometry.projection.detector_to_ray",
"url":67,
"doc":"Find light ray witch intersected on the detector. Bijection of  laueimproc.geometry.projection.ray_to_detector . Parameters      point : torch.Tensor The 2d point in meter in the referencial of the detector of shape (\\ r, \\ p, 2). poni : torch.Tensor The point of normal incidence, callibration parameters according the pyfai convention. Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\\ p', 6). Returns    - ray : torch.Tensor The unitary ray vector of shape (\\ r, \\ broadcast(p, p'), 3).",
"func":1
},
{
"ref":"laueimproc.geometry.projection.ray_to_detector",
"url":67,
"doc":"Find the intersection of the light ray with the detector. Bijection of  laueimproc.geometry.projection.detector_to_ray . Parameters      ray : torch.Tensor The unitary ray vector of shape (\\ r, 3). poni : torch.Tensor The point of normal incidence, callibration parameters according the pyfai convention. Values are [dist, poni_1, poni_2, rot_1, rot_2, rot_3] of shape (\\ p, 6). cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape are (\\ r, \\ p, 2) and (\\ r, \\ p).  False: The final shape are (\\ broadcast(r, p), 2) and broadcast(r, p). Returns    - point : torch.Tensor The 2d points in meter in the referencial of the detector of shape ( ., 2). dist : torch.Tensor The algebrical distance of the ray between the sample and the detector. a positive value means that the beam crashs on the detector. A negative value means it is moving away. The shape is ( .).",
"func":1
},
{
"ref":"laueimproc.geometry.lattice",
"url":68,
"doc":"Link the lattice parameters and the primitive vectors \\(\\mathbf{A}\\)."
},
{
"ref":"laueimproc.geometry.lattice.lattice_to_primitive",
"url":68,
"doc":"Convert the lattice parameters into primitive vectors.  image  / / /build/media/IMGLatticeBc.avif Parameters      lattice : torch.Tensor The array of lattice parameters of shape ( ., 6). Values are \\([a, b, c, \\alpha, \\beta, \\gamma \\) in meters and radians. Returns    - primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) of shape ( ., 3, 3) in the crystal base \\(\\mathcal{B^c}\\). Examples     >>> import torch >>> from laueimproc.geometry.lattice import lattice_to_primitive >>> lattice = torch.tensor([6.0e-10, 3.8e-10, 15e-10, torch.pi/3, torch.pi/2, 2 torch.pi/3]) >>> lattice_to_primitive(lattice) tensor( 6.0000e-10, -1.9000e-10, -6.5567e-17], [ 0.0000e+00, 3.2909e-10, 8.6603e-10], [ 0.0000e+00, 0.0000e+00, 1.2247e-09 ) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.lattice.primitive_to_lattice",
"url":68,
"doc":"Convert the primitive vectors to the lattice parameters. Bijection of  laueimproc.geometry.lattice.lattice_to_primitive .  image  / / /build/media/IMGLattice.avif Parameters      primitive : torch.Tensor Matrix \\(\\mathbf{A}\\) in any orthonormal base. Returns    - lattice : torch.Tensor The array of lattice parameters of shape ( ., 6). Values are \\([a, b, c, \\alpha, \\beta, \\gamma]\\) in meters and radians. Notes   - We have always  primitive_to_lattice(lattice_to_primitive(lattice  lattice , but it is not always the case for the inverse composition because the numerical value of \\(\\mathbf{A}\\) is expressed in the crystal base \\(\\mathcal{B^c}\\). Examples     >>> import torch >>> from laueimproc.geometry.lattice import primitive_to_lattice >>> primitive = torch.tensor( 6.0000e-10, -1.9000e-10, -6.5567e-17],  . [ 0.0000e+00, 3.2909e-10, 8.6603e-10],  . [ 0.0000e+00, 0.0000e+00, 1.2247e-09 ) >>> primitive_to_lattice(primitive)  quartz lattice tensor([6.0000e-10, 3.8000e-10, 1.5000e-09, 1.0472e+00, 1.5708e+00, 2.0944e+00]) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.rotation",
"url":69,
"doc":"Help for making rotation matrix."
},
{
"ref":"laueimproc.geometry.rotation.angle_to_rot",
"url":69,
"doc":"Generate a rotation matrix from the given angles. The rotation are following the pyfai convention. Parameters      theta1 : torch.Tensor or float The first rotation angle of shape (\\ a,). \\( rot_1 = \\begin{pmatrix} 1 & 0 & 0  0 & \\cos(\\theta_1) & -\\sin(\\theta_1)  c & \\sin(\\theta_1) & \\cos(\\theta_1)  \\end{pmatrix} \\) theta2 : torch.Tensor or float The second rotation angle of shape (\\ b,). \\( rot_2 = \\begin{pmatrix} \\cos(\\theta_2) & 0 & \\sin(\\theta_2)  0 & 1 & 0  -\\sin(\\theta_2) & 0 & \\cos(\\theta_2)  \\end{pmatrix} \\) theta3 : torch.Tensor or float The third rotation angle of shape (\\ c,). (inverse of pyfai convention) \\( rot_3 = \\begin{pmatrix} \\cos(\\theta_3) & -\\sin(\\theta_3) & 0  \\sin(\\theta_3) & \\cos(\\theta_3) & 0  0 & 0 & 1  \\end{pmatrix} \\) cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape is (\\ a, \\ b, \\ c, 3, 3).  False: The final shape is (\\ broadcast(a, b, c), 3, 3). Returns    - rot : torch.Tensor The global rotation \\(rot_3 . rot_2 . rot_1\\). Examples     >>> import torch >>> from laueimproc.geometry.rotation import angle_to_rot >>> angle_to_rot(theta1=torch.pi/6, theta2=torch.pi/6, theta3=torch.pi/6) tensor( 0.7500, -0.2165, 0.6250], [ 0.4330, 0.8750, -0.2165], [-0.5000, 0.4330, 0.7500 ) >>> angle_to_rot(torch.randn(4), torch.randn(5, 6), torch.randn(7, 8, 9 .shape torch.Size([4, 5, 6, 7, 8, 9, 3, 3]) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.rotation.rot_to_angle",
"url":69,
"doc":"Extract the rotation angles from a fulle rotation matrix. Bijection of  laueimproc.geometry.rotation.angle_to_rot . Parameters      rot : torch.Tensor The rotation matrix \\(rot_3 . rot_2 . rot_1\\) of shape ( ., 3, 3). Returns    - theta1 : torch.Tensor or float The first rotation angle of shape ( .). \\(\\theta_1 \\in [-\\pi, \\pi]\\) \\( rot_1 = \\begin{pmatrix} 1 & 0 & 0  0 & \\cos(\\theta_1) & -\\sin(\\theta_1)  c & \\sin(\\theta_1) & \\cos(\\theta_1)  \\end{pmatrix} \\) theta2 : torch.Tensor or float The second rotation angle of shape ( .). \\(\\theta_2 \\in [-\\frac{\\pi}{2}, \\frac{\\pi}{2}]\\) \\( rot_2 = \\begin{pmatrix} \\cos(\\theta_2) & 0 & \\sin(\\theta_2)  0 & 1 & 0  -\\sin(\\theta_2) & 0 & \\cos(\\theta_2)  \\end{pmatrix} \\) theta3 : torch.Tensor or float The third rotation angle of shape ( .). (inverse of pyfai convention) \\(\\theta_3 \\in [-\\pi, \\pi]\\) \\( rot_3 = \\begin{pmatrix} \\cos(\\theta_3) & -\\sin(\\theta_3) & 0  \\sin(\\theta_3) & \\cos(\\theta_3) & 0  0 & 0 & 1  \\end{pmatrix} \\) Examples     >>> import torch >>> from laueimproc.geometry.rotation import rot_to_angle >>> rot = torch.tensor( 0.7500, -0.2165, 0.6250],  . [ 0.4330, 0.8750, -0.2165],  . [-0.5000, 0.4330, 0.7500 ) >>> theta1, theta2, theta3 = rot_to_angle(rot) >>> torch.rad2deg(theta1).round() tensor(30.) >>> torch.rad2deg(theta2).round() tensor(30.) >>> torch.rad2deg(theta3).round() tensor(30.) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.rotation.rotate_crystal",
"url":69,
"doc":"Apply an active rotation to the crystal. Parameters      crystal : torch.Tensor The primitive \\ \\mathbf{A})\\) or reciprocal \\ \\mathbf{B})\\) in the base \\(\\mathcal{B}\\). The shape of this parameter is (\\ c, 3, 3). rot : torch.Tensor The active rotation matrix, of shape (\\ r, 3, 3). cartesian_product : boolean, default=True If True (default value), batch dimensions are iterated independently like neasted for loop. Overwise, the batch dimensions are broadcasted like a zip.  True: The final shape is (\\ c, \\ r, 3, 3).  False: The final shape is (\\ broadcast(c, r), 3, 3). Returns    - rotated_crystal: torch.Tensor The batched matricial product  rot @ crystal .",
"func":1
},
{
"ref":"laueimproc.geometry.thetachi",
"url":70,
"doc":"Convertion of diffracted ray to angles. Do not use these functions for intensive calculation, they should only be used for final conversion, not for a calculation step."
},
{
"ref":"laueimproc.geometry.thetachi.thetachi_to_uf",
"url":70,
"doc":"Reconstruct the diffracted ray from the deviation angles. Bijection of  laueimproc.geometry.thetachi.uf_to_thetachi .  image  / / /build/media/IMGThetaChi.avif Parameters      theta : torch.Tensor The half deviation angle in radian, with \\(\\theta \\in [0, \\frac{\\pi}{2}]\\). chi : torch.Tensor The rotation angle in radian from the vertical plan \\ \\mathbf{L_1}, \\mathbf{L_3})\\) to the plan \\ u_i, \\mathcal{B^l_z})\\), with \\(\\chi \\in [-\\pi, \\pi]\\). Returns    - u_f : torch.Tensor The unitary diffracted ray of shape ( broadcast(theta.shape, chi.shape), 3). It is expressed in the lab base \\(\\mathcal{B^l}\\). Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).  This function is slow, use  laueimproc.geometry.bragg.uq_to_uf if you can. Examples     >>> import torch >>> from laueimproc.geometry.thetachi import thetachi_to_uf >>> theta = torch.deg2rad(torch.tensor([15., 30., 45.] >>> chi = torch.deg2rad(torch.tensor([ -0., 90., -45.] >>> thetachi_to_uf(theta, chi).round(decimals=4) tensor( 0.5000, 0.0000, 0.8660], [-0.0000, -0.8660, 0.5000], [ 0.7071, 0.7071, -0.0000 ) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.thetachi.uf_to_thetachi",
"url":70,
"doc":"Find the angular deviation of the dffracted ray. Bijection of  laueimproc.geometry.thetachi.thetachi_to_uf .  image  / / /build/media/IMGThetaChi.avif Parameters      u_f : torch.Tensor The unitary diffracted ray of shape ( ., 3) in the lab base \\(\\mathcal{B^l}\\). Returns    - theta : torch.Tensor The half deviation angle in radian, of shape ( .). \\(\\theta \\in [0, \\frac{\\pi}{2}]\\) chi : torch.Tensor The counterclockwise (trigonometric) rotation of the diffracted ray if you look as u_i. It is the angle from the vertical plan \\ \\mathbf{L_1}, \\mathbf{L_3})\\) to the plan \\ u_i, \\mathcal{B^l_z})\\). The shape is ( .) as well. \\(\\chi \\in [-\\pi, \\pi]\\) Notes   -  According the pyfai convention, \\(u_i = \\mathbf{L_1}\\).  This function is slow, use  laueimproc.geometry.bragg.uf_to_uq if you can. Examples     >>> import torch >>> from laueimproc.geometry.thetachi import uf_to_thetachi >>> u_f = torch.tensor( 1/2, 0, 3 (1/2)/2], [0, -3 (1/2)/2, 1/2], [2 (1/2), 2 (1/2), 0 ) >>> theta, chi = uf_to_thetachi(u_f) >>> torch.rad2deg(theta).round() tensor([15., 30., 45.]) >>> torch.rad2deg(chi).round() tensor([ -0., 90., -45.]) >>>",
"func":1
},
{
"ref":"laueimproc.geometry.hkl",
"url":71,
"doc":"Help to select the hkl indices."
},
{
"ref":"laueimproc.geometry.hkl.select_hkl",
"url":71,
"doc":"Reject the hkl sistematicaly out of energy band. Parameters      reciprocal : torch.Tensor, optional Matrix \\(\\mathbf{B}\\) in any orthonormal base. max_hkl : int, optional The maximum absolute hkl sum such as |h| + |k| + |l|  >> import torch >>> from laueimproc.geometry.hkl import select_hkl >>> reciprocal = torch.tensor( 2.7778e9, 0, 0],  . [1.2142e2, 2.7778e9, 0],  . [1.2142e2, 1.2142e2, 2.7778e9 ) >>> select_hkl(max_hkl=18) tensor( 0, -18, 0], [ 0, -17, -1], [ 0, -17, 0],  ., [ 17, 0, 1], [ 17, 1, 0], [ 18, 0, 0 , dtype=torch.int16) >>> len(_) 4578 >>> select_hkl(max_hkl=18, keep_harmonics=False) tensor( 0, -17, -1], [ 0, -17, 1], [ 0, -16, -1],  ., [ 17, 0, -1], [ 17, 0, 1], [ 17, 1, 0 , dtype=torch.int16) >>> len(_) 3661 >>> select_hkl(reciprocal, e_max=20e3  1.60e-19)  20 keV tensor( 0, -11, -3], [ 0, -11, -2], [ 0, -11, -1],  ., [ 11, 3, 0], [ 11, 3, 1], [ 11, 3, 2 , dtype=torch.int16) >>> len(_) 1463 >>> select_hkl(reciprocal, e_max=20e3  1.60e-19, keep_harmonics=False)  20 keV tensor( 0, -11, -1], [ 0, -11, 1], [ 0, -10, -1],  ., [ 11, 0, -1], [ 11, 0, 1], [ 11, 1, 0 , dtype=torch.int16) >>> len(_) 1149 >>>",
"func":1
},
{
"ref":"laueimproc.geometry.model",
"url":72,
"doc":"Full model for complete simulation. Combines the various elementary functions into a torch module."
},
{
"ref":"laueimproc.geometry.model.GeneralSimulator",
"url":72,
"doc":"General simulation class. Attributes      lattice : None | torch.Tensor The lattice parameters of shape ( ., 6). phi : None | tuple[torch.Tensor, torch.Tensor, torch.Tensor] The decomposition of rot in elementary angles. rot : None | torch.Tensor The rotation matrix of shape ( ., 3, 3). Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"laueimproc.geometry.model.GeneralSimulator.lattice",
"url":72,
"doc":"Return the lattice parameters of shape ( ., 6)."
},
{
"ref":"laueimproc.geometry.model.GeneralSimulator.phi",
"url":72,
"doc":"Return the decomposition of rot in elementary angles."
},
{
"ref":"laueimproc.geometry.model.GeneralSimulator.rot",
"url":72,
"doc":"Return the rotation matrix of shape ( ., 3, 3)."
},
{
"ref":"laueimproc.geometry.model.SimulatorUf2Cam",
"url":72,
"doc":"Simulation from u_f to camera. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"laueimproc.geometry.model.SimulatorUf2Cam.u_f",
"url":72,
"doc":"Return the \\(u_f\\) vector."
},
{
"ref":"laueimproc.geometry.model.SimulatorUf2Cam.forward",
"url":72,
"doc":"Simulate.",
"func":1
},
{
"ref":"laueimproc.geometry.model.SimulatorUf2Cam.lattice",
"url":72,
"doc":"Return the lattice parameters of shape ( ., 6)."
},
{
"ref":"laueimproc.geometry.model.SimulatorUf2Cam.phi",
"url":72,
"doc":"Return the decomposition of rot in elementary angles."
},
{
"ref":"laueimproc.geometry.model.SimulatorUf2Cam.rot",
"url":72,
"doc":"Return the rotation matrix of shape ( ., 3, 3)."
},
{
"ref":"laueimproc.geometry.model.Simulator",
"url":72,
"doc":"Simulate a full multigrain laue diagram."
},
{
"ref":"laueimproc.geometry.model.FullSimulator",
"url":72,
"doc":"Full simulation from lattice parameters to final laue diagram. Initialise the model. Parameters      lattice : torch.Tensor, optional The lattice parameters of the grains to be simulated, shape ( ., 6). rot : torch.Tensor, optional The rotation matrix of the different grains, shape ( ., 3, 3). poni : torch.Tensor, optional The camera calibration parameters, shape ( ., 6)."
},
{
"ref":"laueimproc.geometry.model.FullSimulator.forward",
"url":72,
"doc":"Simulate the grains. Examples     >>> import torch >>> from laueimproc.geometry.model import FullSimulator >>> lattice = torch.tensor([3.6e-10, 3.6e-10, 3.6e-10, torch.pi/2, torch.pi/2, torch.pi/2]) >>> poni = torch.tensor([0.07, 73.4e-3, 73.4e-3, 0.0, -torch.pi/2, 0.0]) >>> rot = torch.eye(3) >>> simulator = FullSimulator(lattice, rot, poni) >>> simulator()",
"func":1
},
{
"ref":"laueimproc.convention",
"url":73,
"doc":"Provide tools for switching convention."
},
{
"ref":"laueimproc.convention.det_to_poni",
"url":73,
"doc":"Convert a .det config into a .poni config. Bijection of  laueimproc.convention.poni_to_det . Parameters      det : torch.Tensor The 5 + 1 ordered .det calibration parameters as a tensor of shape ( ., 6):   dd : The distance sample to the point of normal incidence in mm.   xcen : The x coordinate of point of normal incidence in pixel.   ycen : The y coordinate of point of normal incidence in pixel.   xbet : One of the angle of rotation in degrees.   xgam : The other angle of rotation in degrees.   pixelsize : The size of one pixel in mm. Returns    - poni : torch.Tensor The 6 ordered .poni calibration parameters as a tensor of shape ( ., 6):   dist : The distance sample to the point of normal incidence in m.   poni1 : The coordinate of the point of normal incidence along d1 in m.   poni2 : The coordinate of the point of normal incidence along d2 in m.   rot1 : The first rotation in radian.   rot2 : The second rotation in radian.   rot1 : The third rotation in radian. Examples     >>> import torch >>> from laueimproc.convention import det_to_poni, lauetools_to_pyfai, or_to_lauetools >>> from laueimproc.geometry.projection import ray_to_detector >>> uf_or = torch.randn(1000, 3) >>> uf_or  = torch.rsqrt(uf_or.sum(dim=1, keepdim=True >>> det = torch.tensor([77.0, 800.0, 1200.0, 20.0, 20.0, 0.08]) >>> >>>  using lauetools >>> uf_pyfai = lauetools_to_pyfai(or_to_lauetools(uf_or >>> poni = det_to_poni(det) >>> xy_improc, _ = ray_to_detector(uf_pyfai, poni) >>> x_improc, y_improc = xy_improc[:, 0], xy_improc[:, 1] >>> cond = (x_improc > -1) & (x_improc  -1) & (y_improc  >> x_improc, y_improc = x_improc[cond], y_improc[cond] >>> >>>  using laueimproc >>> try:  . from LaueTools.LaueGeometry import calc_xycam  . except ImportError:  . pass  . else:  . x_tools, y_tools, _ = calc_xycam(  . uf_or.numpy(force=True), det[:-1].numpy(force=True), pixelsize=float(det[5])  . )  . x_tools = torch.asarray(x_tools, dtype=x_improc.dtype)[cond]  . y_tools = torch.asarray(y_tools, dtype=y_improc.dtype)[cond]  . x_tools, y_tools = det[5]  1e-3  x_tools, det[5]  1e-3  y_tools  . assert torch.allclose(x_improc, x_tools)  . assert torch.allclose(y_improc, y_tools)  . >>>",
"func":1
},
{
"ref":"laueimproc.convention.ij_to_xy",
"url":73,
"doc":"Switch the axis i and j, and append 1/2 to all values.   ij : Extension by continuity (N -> R) of the numpy convention (height, width). The first axis iterates on lines from top to bottom, the second on columns from left to right. The origin (i=0, j=0) correspond to the top left image corner of the top left pixel. It means that the center of the top left pixel has the coordinate (i=1/2, j=1/2).   xy : A transposition and a translation of the origin of the  ij convention. The first axis iterates on columns from left to right, the second on lines from top to bottom. In an image, the point (x=1, y=1) correspond to the middle of the top left pixel.  image  / /build/media/IMGConvIJXY.avif Parameters      array : torch.Tensor or np.ndarray The data in ij convention. i, j : tuple, int, slice or Ellipsis The indexing of the i subdata and j subdata. Returns    - array : torch.Tensor or np.ndarray A reference to the ij_array, with the axis converted in xy convention. Notes   - Input and output data are shared in place. Examples     >>> import torch >>> from laueimproc.convention import ij_to_xy >>> array = torch.zeros 10, 2 >>> array[:, 0] = torch.linspace(0, 1, 10)  i axis >>> array[:, 1] = torch.linspace(2, 1, 10)  j axis >>> array tensor( 0.0000, 2.0000], [0.1111, 1.8889], [0.2222, 1.7778], [0.3333, 1.6667], [0.4444, 1.5556], [0.5556, 1.4444], [0.6667, 1.3333], [0.7778, 1.2222], [0.8889, 1.1111], [1.0000, 1.0000 ) >>> ij_to_xy(array, i=( ., 0), j=( ., 1 tensor( 2.5000, 0.5000], [2.3889, 0.6111], [2.2778, 0.7222], [2.1667, 0.8333], [2.0556, 0.9444], [1.9444, 1.0556], [1.8333, 1.1667], [1.7222, 1.2778], [1.6111, 1.3889], [1.5000, 1.5000 ) >>> _ is array  inplace True >>>",
"func":1
},
{
"ref":"laueimproc.convention.ij_to_xy_decorator",
"url":73,
"doc":"Append the argument conv to a function to allow user switching convention.",
"func":1
},
{
"ref":"laueimproc.convention.lauetools_to_or",
"url":73,
"doc":"Active convertion of the vectors from lauetools base to odile base. Bijection of  laueimproc.convention.or_to_lauetools .  image  / /build/media/IMGLauetoolsOr.avif Parameters      vect_lauetools : torch.Tensor The vector of shape ( ., 3,  .) in the lauetools orthonormal base. dim : int, default=-1 The axis index of the non batch dimension, such that vect.shape[dim] = 3. Returns    - vect_or : torch.Tensor The input vect, with the axis converted in the odile base. Examples     >>> import torch >>> from laueimproc.convention import lauetools_to_or, or_to_lauetools >>> vect_odile = torch.tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>> lauetools_to_or(vect_odile, dim=0) tensor( 0, -1, 0, -1], [ 1, 0, 0, 1], [ 0, 0, 1, 1 ) >>> or_to_lauetools(_, dim=0) tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>>",
"func":1
},
{
"ref":"laueimproc.convention.lauetools_to_pyfai",
"url":73,
"doc":"Active convertion of the vectors from lauetools base to pyfai base. Bijection of  laueimproc.convention.pyfai_to_lauetools .  image  / /build/media/IMGPyfaiLauetools.avif Parameters      vect_lauetools : torch.Tensor The vector of shape ( ., 3,  .) in the lauetools orthonormal base. dim : int, default=-1 The axis index of the non batch dimension, such that vect.shape[dim] = 3. Returns    - vect_pyfai : torch.Tensor The input vect, with the axis converted in the pyfai base. Examples     >>> import torch >>> from laueimproc.convention import lauetools_to_pyfai, pyfai_to_lauetools >>> vect_pyfai = torch.tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>> lauetools_to_pyfai(vect_pyfai, dim=0) tensor( 0, 0, 1, 1], [ 0, -1, 0, -1], [ 1, 0, 0, 1 ) >>> pyfai_to_lauetools(_, dim=0) tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>>",
"func":1
},
{
"ref":"laueimproc.convention.or_to_lauetools",
"url":73,
"doc":"Active convertion of the vectors from odile base to lauetools base. Bijection of  laueimproc.convention.lauetools_to_or .  image  / /build/media/IMGLauetoolsOr.avif Parameters      vect_or : torch.Tensor The vector of shape ( ., 3,  .) in the odile orthonormal base. dim : int, default=-1 The axis index of the non batch dimension, such that vect.shape[dim] = 3. Returns    - vect_lauetools : torch.Tensor The input vect, with the axis converted in the lauetools base. Examples     >>> import torch >>> from laueimproc.convention import lauetools_to_or, or_to_lauetools >>> vect_or = torch.tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>> or_to_lauetools(vect_or, dim=0) tensor( 0, 1, 0, 1], [-1, 0, 0, -1], [ 0, 0, 1, 1 ) >>> lauetools_to_or(_, dim=0) tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>>",
"func":1
},
{
"ref":"laueimproc.convention.poni_to_det",
"url":73,
"doc":"Convert a .det config into a .poni config. Bijection of  laueimproc.convention.det_to_poni . Parameters      poni : torch.Tensor The 6 ordered .poni calibration parameters as a tensor of shape ( ., 6):   dist : The distance sample to the point of normal incidence in m.   poni1 : The coordinate of the point of normal incidence along d1 in m.   poni2 : The coordinate of the point of normal incidence along d2 in m.   rot1 : The first rotation in radian.   rot2 : The second rotation in radian.   rot1 : The third rotation in radian. pixelsize : torch.Tensor The size of one pixel in mm of shape ( .). Returns    - det : torch.Tensor The 5 + 1 ordered .det calibration parameters as a tensor of shape ( ., 6):   dd : The distance sample to the point of normal incidence in mm.   xcen : The x coordinate of point of normal incidence in pixel.   ycen : The y coordinate of point of normal incidence in pixel.   xbet : One of the angle of rotation in degrees.   xgam : The other angle of rotation in degrees.   pixelsize : The size of one pixel in mm. Examples     >>> import torch >>> from laueimproc.convention import det_to_poni, poni_to_det >>> det = torch.tensor([77.0, 800.0, 1200.0, 20.0, 20.0, 0.08]) >>> torch.allclose(det, poni_to_det(det_to_poni(det), det[5] True >>>",
"func":1
},
{
"ref":"laueimproc.convention.pyfai_to_lauetools",
"url":73,
"doc":"Active convertion of the vectors from pyfai base to lauetools base. Bijection of  laueimproc.convention.lauetools_to_pyfai .  image  / /build/media/IMGPyfaiLauetools.avif Parameters      vect_pyfai : torch.Tensor The vector of shape ( ., 3,  .) in the pyfai orthonormal base. dim : int, default=-1 The axis index of the non batch dimension, such that vect.shape[dim] = 3. Returns    - vect_lauetools : torch.Tensor The input vect, with the axis converted in the lauetools base. Examples     >>> import torch >>> from laueimproc.convention import lauetools_to_pyfai, pyfai_to_lauetools >>> vect_pyfai = torch.tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>> pyfai_to_lauetools(vect_pyfai, dim=0) tensor( 0, 0, 1, 1], [ 0, -1, 0, -1], [ 1, 0, 0, 1 ) >>> lauetools_to_pyfai(_, dim=0) tensor( 1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1 ) >>>",
"func":1
}
]