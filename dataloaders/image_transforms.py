"""
Image transforms functions for data augmentation
Credit to Dr. Jo Schlemper
"""

from collections import Sequence
import cv2
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from numpy.lib.stride_tricks import as_strided

###### UTILITIES ######
def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        #print(config)
        raise Exception('unsupported format')
    return ret

def get_translation_matrix(translation):
    """ translation: [tx, ty] """
    tx, ty = translation
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    return translation_matrix



def get_rotation_matrix(rotation, input_shape, centred=True):
    theta = np.pi / 180 * np.array(rotation)
    if centred:
        rotation_matrix = cv2.getRotationMatrix2D((input_shape[0]/2, input_shape[1]//2), rotation, 1)
        rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
    else:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0, 0, 1]])
    return rotation_matrix

def get_zoom_matrix(zoom, input_shape, centred=True):
    zx, zy = zoom
    if centred:
        zoom_matrix = cv2.getRotationMatrix2D((input_shape[0]/2, input_shape[1]//2), 0, zoom[0])
        zoom_matrix = np.vstack([zoom_matrix, [0, 0, 1]])
    else:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0,  1]])
    return zoom_matrix

def get_shear_matrix(shear_angle):
    theta = (np.pi * shear_angle) / 180
    shear_matrix = np.array([[1, -np.sin(theta), 0],
                             [0,  np.cos(theta), 0],
                             [0, 0, 1]])
    return shear_matrix

###### AFFINE TRANSFORM ######
class RandomAffine(object):
    """Apply random affine transformation on a numpy.ndarray (H x W x C)
    Comment by co1818: this is still doing affine on 2d (H x W plane).
                        A same transform is applied to all C channels

    Parameter:
    ----------

    alpha: Range [0, 4] seems good for small images

    order: interpolation method (c.f. opencv)
    """

    def __init__(self,
                 rotation_range=None,
                 translation_range=None,
                 shear_range=None,
                 zoom_range=None,
                 zoom_keep_aspect=False,
                 interp='bilinear',
                 order=3):
        """
        Perform an affine transforms.

        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated randomly between (-degrees, degrees)

        translation_range : (x_shift, y_shift)
            shifts in pixels

        *NOT TESTED* shear_range : float
            image will be sheared randomly between (-degrees, degrees)

        zoom_range : (zoom_min, zoom_max)
            list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom.
            Anything less than 1.0 will zoom in on the image,
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in,
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        """

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.zoom_keep_aspect = zoom_keep_aspect
        self.interp = interp
        self.order = order

    def build_M(self, input_shape):
        tfx = []
        final_tfx = np.eye(3)
        if self.rotation_range:
            rot = np.random.uniform(-self.rotation_range, self.rotation_range)
            tfx.append(get_rotation_matrix(rot, input_shape))
        if self.translation_range:
            tx = np.random.uniform(-self.translation_range[0], self.translation_range[0])
            ty = np.random.uniform(-self.translation_range[1], self.translation_range[1])
            tfx.append(get_translation_matrix((tx,ty)))
        if self.shear_range:
            rot = np.random.uniform(-self.shear_range, self.shear_range)
            tfx.append(get_shear_matrix(rot))
        if self.zoom_range:
            sx = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            if self.zoom_keep_aspect:
                sy = sx
            else:
                sy = np.random.uniform(self.zoom_range[0], self.zoom_range[1])

            tfx.append(get_zoom_matrix((sx, sy), input_shape))

        for tfx_mat in tfx:
            final_tfx = np.dot(tfx_mat, final_tfx)

        return final_tfx.astype(np.float32)

    def __call__(self, image):
        # build matrix
        input_shape = image.shape[:2]
        M = self.build_M(input_shape)

        res = np.zeros_like(image)
        #if isinstance(self.interp, Sequence):
        if type(self.order) is list or type(self.order) is tuple:
            for i, intp in enumerate(self.order):
                res[..., i] = affine_transform_via_M(image[..., i], M[:2], interp=intp)
        else:
            # squeeze if needed
            orig_shape = image.shape
            image_s = np.squeeze(image)
            res = affine_transform_via_M(image_s, M[:2], interp=self.order)
            res = res.reshape(orig_shape)

            #res = affine_transform_via_M(image, M[:2], interp=self.order)

        return res

def affine_transform_via_M(image, M, borderMode=cv2.BORDER_CONSTANT, interp=cv2.INTER_NEAREST):
    imshape = image.shape
    shape_size = imshape[:2]

    # Random affine
    warped = cv2.warpAffine(image.reshape(shape_size + (-1,)), M, shape_size[::-1],
                            flags=interp, borderMode=borderMode)

    #print(imshape, warped.shape)

    warped = warped[..., np.newaxis].reshape(imshape)

    return warped

###### ELASTIC TRANSFORM ######
def elastic_transform(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result


def elastic_transform_nd(image, alpha, sigma, random_state=None, order=1, lazy=False):
    """Expects data to be (nx, ny, n1 ,..., nm)
    params:
    ------

    alpha:
    the scaling parameter.
    E.g.: alpha=2 => distorts images up to 2x scaling

    sigma:
    standard deviation of gaussian filter.
    E.g.
         low (sig~=1e-3) => no smoothing, pixelated.
         high (1/5 * imsize) => smooth, more like affine.
         very high (1/2*im_size) => translation
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    imsize = shape[:2]
    dim = shape[2:]

    # Random affine
    blur_size = int(4*sigma) | 1
    dx = cv2.GaussianBlur(random_state.rand(*imsize)*2-1,
                          ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur(random_state.rand(*imsize)*2-1,
                          ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    # use as_strided to copy things over across n1...nn channels
    dx = as_strided(dx.astype(np.float32),
                    strides=(0,) * len(dim) + (4*shape[1], 4),
                    shape=dim+(shape[0], shape[1]))
    dx = np.transpose(dx, axes=(-2, -1) + tuple(range(len(dim))))

    dy = as_strided(dy.astype(np.float32),
                    strides=(0,) * len(dim) + (4*shape[1], 4),
                    shape=dim+(shape[0], shape[1]))
    dy = np.transpose(dy, axes=(-2, -1) + tuple(range(len(dim))))

    coord = np.meshgrid(*[np.arange(shape_i) for shape_i in (shape[1], shape[0]) + dim])
    indices = [np.reshape(e+de, (-1, 1)) for e, de in zip([coord[1], coord[0]] + coord[2:],
                                                          [dy, dx] + [0] * len(dim))]

    if lazy:
        return indices

    return map_coordinates(image, indices, order=order, mode='reflect').reshape(shape)

class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma, order=1):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order

    def __call__(self, image):
        if isinstance(self.alpha, Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform_nd(image, alpha=alpha, sigma=sigma, order=self.order)

class RandomFlip3D(object):

    def __init__(self, h=True, v=True, t=True, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p

        v : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.depth = t
        self.p = p

    def __call__(self, x, y=None):
        # horizontal flip with p = self.p
        if self.horizontal:
            if np.random.random() < self.p:
                x = x[::-1, ...]

        # vertical flip with p = self.p
        if self.vertical:
            if np.random.random() < self.p:
                x = x[:, ::-1, ...]

        if self.depth:
            if np.random.random() < self.p:
                x = x[..., ::-1]

        return x


