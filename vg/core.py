import math
import numpy as np
from ._helpers import _check_value_any, broadcast_and_tile, raise_dimension_error
from .shape import check, check_value_any

__all__ = [
    "normalize",
    "perpendicular",
    "project",
    "scalar_projection",
    "reject",
    "reject_axis",
    "magnitude",
    "euclidean_distance",
    "angle",
    "signed_angle",
    "rotate",
    "scale_factor",
    "orient",
    "almost_zero",
    "almost_unit_length",
    "almost_collinear",
    "almost_equal",
    "principal_components",
    "major_axis",
    "apex",
    "apex_and_opposite",
    "argapex",
    "nearest",
    "farthest",
    "basis",
    "within",
    "average",
    "cross",
    "dot",
]


def normalize(vector):
    """
    Return the vector, normalized.

    If vector is 2d, treats it as stacked vectors, and normalizes each one.

    """
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector)
    elif vector.ndim == 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        raise_dimension_error(vector)


def perpendicular(v1, v2, normalized=True):
    """
    Given two noncollinear vectors, return a vector perpendicular to both.

    Result vectors follow the right-hand rule. When the right index finger
    points along `v1` and the right middle finger along `v2`, the right thumb
    points along the result.

    When one or both sets of inputs is stacked, compute the perpendicular
    vectors elementwise, returning a stacked result. (e.g. when `v1` and `v2`
    are both stacked, `result[k]` is perpendicular to `v1[k]` and `v2[k]`.)

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors. If
            stacked, the shape must be the same as `v1`.
        normalized (bool): When `True`, the result vector is guaranteed to be
            unit length.

    Return:
        np.arraylike: An array with the same shape as `v1` and `v2`.

    See also:
        - https://en.wikipedia.org/wiki/Cross_product#Definition
        - https://commons.wikimedia.org/wiki/File:Right_hand_rule_cross_product.svg
    """
    result = cross(v1, v2)
    return normalize(result) if normalized else result


def project(vector, onto):
    """
    Compute the vector projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if vector.ndim == 1:
        return scalar_projection(vector, onto=onto) * normalize(onto)
    elif vector.ndim == 2:
        return scalar_projection(vector, onto=onto)[:, np.newaxis] * normalize(onto)
    else:
        raise_dimension_error(vector)


def scalar_projection(vector, onto):
    """
    Compute the scalar projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if vector.ndim == 1:
        check(locals(), "vector", (3,))
        check(locals(), "onto", (3,))
    else:
        k = check(locals(), "vector", (-1, 3))
        if onto.ndim == 1:
            check(locals(), "onto", (3,))
        else:
            check(locals(), "onto", (k, 3))

    return dot(vector, normalize(onto))


def reject(vector, from_v):
    """
    Compute the vector rejection of `vector` from `from_v` -- i.e.
    the vector component of `vector` perpendicular to `from_v`.

    `from_v` need not be normalized.

    """
    return vector - project(vector, onto=from_v)


def reject_axis(vector, axis, squash=False):
    """
    Compute the vector component of `vector` perpendicular to the basis
    vector specified by `axis`. 0 means x, 1 means y, 2 means z.

    In other words, return a copy of vector that zeros the `axis` component.

    When `squash` is True, instead of zeroing the component, it drops it, so
    an input vector (in R3) is mapped to a point in R2.

    (N.B. Don't be misled: this meaning of `axis` is pretty different from
    the typical meaning in numpy.)

    """
    if squash:
        dims_to_keep = [0, 1, 2]
        try:
            dims_to_keep.remove(axis)
        except ValueError:
            raise ValueError("axis should be 0, 1, or 2")

        if vector.ndim == 1:
            return vector[dims_to_keep]
        elif vector.ndim == 2:
            return vector[:, dims_to_keep]
        else:
            raise_dimension_error(vector)
    else:
        if axis not in [0, 1, 2]:
            raise ValueError("axis should be 0, 1, or 2")
        result = vector.copy()
        if vector.ndim == 1:
            result[axis] = 0.0
        elif vector.ndim == 2:
            result[:, axis] = 0.0
        else:
            raise_dimension_error(vector)
        return result


def magnitude(vector):
    """
    Compute the magnitude of `vector`. For a stacked input, compute the
    magnitude of each one.

    Args:
        vector (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.

    Returns:
        object: For a `(3,)` input, a `float` with the magnitude. For a `kx3`
            input, a `(k,)` array.
    """
    if vector.ndim == 1:
        return np.linalg.norm(vector)
    elif vector.ndim == 2:
        return np.linalg.norm(vector, axis=1)
    else:
        raise_dimension_error(vector)


# Alias because angle()'s parameter shadows the name.
_normalize = normalize


def euclidean_distance(v1, v2):
    """
    Compute Euclidean distance, which is the distance between two points in a
    straight line. This can be done individually by passing in single
    point for either or both arguments, or pairwise by passing in stacks of
    points.

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors. If
            stacks are provided for both `v1` and `v2` they must have the
            same shape.

    Returns:
        object: When both inputs are `(3,)`, a `float` with the distance.
        Otherwise a `(k,)` array.
    """
    k = check_value_any(v1, (3,), (-1, 3), name="v1")
    check_value_any(
        v2,
        (3,),
        (-1 if k is None else k, 3),
        name="v2",
    )

    if v1.ndim == 1 and v2.ndim == 1:
        return np.sqrt(np.sum(np.square(v2 - v1)))
    else:
        return np.sqrt(np.sum(np.square(v2 - v1), axis=1))


def angle(v1, v2, look=None, assume_normalized=False, units="deg"):
    """
    Compute the unsigned angle between two vectors. For a stacked input, the
    angle is computed pairwise.

    When `look` is provided, the angle is computed in that viewing plane
    (`look` is the normal). Otherwise the angle is computed in 3-space.

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A vector or stack of vectors with the same shape as
            `v1`.
        look (np.arraylike): A `(3,)` vector specifying the normal of a viewing
            plane, or `None` to compute the angle in 3-space.
        assume_normalized (bool): When `True`, assume the input vectors
            are unit length. This improves performance, however when the inputs
            are not normalized, setting this will cause an incorrect results.
        units (str): `'deg'` to return degrees or `'rad'` to return radians.

    Return:
        object: For a `(3,)` input, a `float` with the angle. For a `kx3`
        input, a `(k,)` array.
    """
    if units not in ["deg", "rad"]:
        raise ValueError("Unrecognized units {}; expected deg or rad".format(units))

    if look is not None:
        # This is a simple approach. Since this is working in two dimensions,
        # a smarter approach could reduce the amount of computation needed.
        v1, v2 = [reject(v, from_v=look) for v in (v1, v2)]

    dot_products = np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))

    if assume_normalized:
        cosines = dot_products
    else:
        cosines = dot_products / magnitude(v1) / magnitude(v2)

    # Clip, because the dot product can slip past 1 or -1 due to rounding and
    # we can't compute arccos(-1.00001).
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    if units == "deg":
        angles = np.degrees(angles)

    return angles[0] if v1.ndim == 1 and v2.ndim == 1 else angles


def signed_angle(v1, v2, look, units="deg"):
    """
    Compute the signed angle between two vectors. For a stacked input, the
    angle is computed pairwise.

    Results are in the range -180 and 180 (or `-math.pi` and `math.pi`). A
    positive number indicates a clockwise sweep from `v1` to `v2`. A negative
    number is counterclockwise.

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A vector or stack of vectors with the same shape as
            `v1`.
        look (np.arraylike): A `(3,)` vector specifying the normal of the
            viewing plane.
        units (str): `'deg'` to return degrees or `'rad'` to return radians.

    Returns:
        object: For a `(3,)` input, a `float` with the angle. For a `kx3`
        input, a `(k,)` array.
    """
    # The sign of (A x B) dot look gives the sign of the angle.
    # > 0 means clockwise, < 0 is counterclockwise.
    sign = np.array(np.sign(np.cross(v1, v2).dot(look)))

    # 0 means collinear: 0 or 180. Let's call that clockwise.
    sign[sign == 0] = 1

    return sign * angle(v1, v2, look, units=units)


def rotate(vector, around_axis, angle, units="deg", assume_normalized=False):
    """
    Rotate a point or vector around a given axis. The direction of rotation
    around `around_axis` is determined by the right-hand rule.

    Args:
        vector (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        around_axis (np.arraylike): A `(3,)` vector specifying the axis of rotation.
        assume_normalized (bool): When `True`, assume `around_axis` is unit
            length. This improves performance marginally, however
            when the inputs are not normalized, setting this will cause an
            incorrect results.
        units (str): `'deg'` to specify `angle` in degrees or `'rad'` to specify
            radians.

    Returns:
        np.arraylike: The transformed point or points. This has the same shape as
            `vector`.

    See also:
        - https://en.wikipedia.org/wiki/Cross_product#Definition
        - https://commons.wikimedia.org/wiki/File:Right_hand_rule_cross_product.svg
    """
    if units == "deg":
        angle = math.radians(angle)
    elif units != "rad":
        raise ValueError('Unknown units "{}"; expected "deg" or "rad"'.format(units))

    cosine = math.cos(angle)
    sine = math.sin(angle)

    if not assume_normalized:
        around_axis = normalize(around_axis)

    if vector.ndim == 1:
        dot_products = np.inner(around_axis, vector)
    elif vector.ndim == 2:
        dot_products = np.inner(around_axis, vector)[:, np.newaxis]
    else:
        raise_dimension_error(vector)

    # Rodrigues' rotation formula.
    return (
        cosine * vector
        + sine * np.cross(around_axis, vector)
        + (1 - cosine) * dot_products * around_axis
    )


def scale_factor(v1, v2):
    """
    Given two parallel vectors, compute the scale factor `k` such that
    `k * v1` is approximately equal to `v2`.

    Args:
        v1 (np.arraylike): A vector in `R^3` or a `kx3` stack of vectors.
        v2 (np.arraylike): A second vector in `R^3` or a `kx3` stack of
          vectors. If `v1` and `v2` are both stacked, they must be the
          same shape.

    Returns:
        object: A float containing the scale factor `k`, or `nan` if `v1`
        is the zero vector. If either input is stacked, the result will also
        be stacked.
    """
    k = _check_value_any(v1, (3,), (-1, 3), name="v1")
    _check_value_any(v2, (3,), (-1 if k is None else k, 3), name="v1")

    v1_dot_v2 = dot(v1, v2)
    v1_dot_v1 = dot(v1, v1)

    if np.isscalar(v1_dot_v1) and v1_dot_v1 == 0:
        v1_dot_v1 = np.nan
    elif not np.isscalar(v1_dot_v1):
        v1_dot_v1[v1_dot_v1 == 0] = np.nan

    return v1_dot_v2 / v1_dot_v1


def orient(vector, along, reverse=False):
    """
    Given two vectors, flip the first if necessary, so that it points
    (approximately) along the second vector rather than (approximately)
    opposite it.

    Args:
        vector (np.arraylike): A vector in `R^3`.
        along (np.arraylike): A second vector in `R^3`.
        reverse (bool): When `True`, reverse the logic, returning a vector
          that points against `along`.

    Returns:
        np.arraylike: Either `vector` or `-vector`.
    """
    check(locals(), "vector", (3,))
    check(locals(), "along", (3,))

    projected = project(vector, onto=along)
    computed_scale_factor = scale_factor(projected, along)
    if not reverse and computed_scale_factor < 0:
        return -vector
    elif reverse and computed_scale_factor > 0:
        return -vector
    else:
        return vector


def almost_zero(v, atol=1e-08):
    """
    Test if v is almost the zero vector.

    """
    return np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=atol)


def almost_unit_length(vector, atol=1e-08):
    """
    Test if the `vector` has almost unit length. For a stacked input, test each
    one.

    Args:
        vector (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.

    Returns:
        object: For a `(3,)` input, a `bool`. For a `kx3` input, a `(k,)`
        array.
    """
    return np.isclose(magnitude(vector), 1.0, rtol=0, atol=atol)


def almost_collinear(v1, v2, atol=1e-08):
    """
    Test if `v1` and `v2` are almost collinear.

    This will return true if either `v1` or `v2` is the zero vector, because
    mathematically speaking, the zero vector is collinear to everything.

    Geometrically that doesn't necessarily make sense, so if you want to handle
    zero vectors specially, you can test your inputs with `vg.almost_zero()`.
    """
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    return np.isclose(norm, 0.0, rtol=0, atol=atol)


def almost_equal(v1, v2, atol=1e-08):
    """
    Test if `v1` and `v2` are equal within the given absolute tolerance.

    See also:
        - https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html

    """
    return np.allclose(v1, v2, rtol=0, atol=atol)


def principal_components(coords):
    """
    Compute the principal components of the input coordinates. These are
    useful for dimensionality reduction and feature modeling.

    Args:
        coords (np.arraylike): A `nxk` stack of coordinates.

    Returns:
        np.ndarray: A `kxk` stack of vectors.

    See also:
        - http://setosa.io/ev/principal-component-analysis/
        - https://en.wikipedia.org/wiki/Principal_component_analysis
        - https://plot.ly/ipython-notebooks/principal-component-analysis/
    """
    mean = np.mean(coords, axis=0)
    _, _, result = np.linalg.svd(coords - mean)
    return result


def major_axis(coords):
    """
    Compute the first principal component of the input coordinates. This is
    the vector which best describes the multidimensional data using a single
    dimension.

    Args:
        coords (np.arraylike): A `nxk` stack of coordinates.

    Returns:
        np.ndarray: A `(k,)` vector.

    See also:
        - http://setosa.io/ev/principal-component-analysis/
        - https://en.wikipedia.org/wiki/Principal_component_analysis
        - https://plot.ly/ipython-notebooks/principal-component-analysis/
    """
    return principal_components(coords)[0]


def argapex(points, along):
    """
    Find the index of the most extreme point in the direction provided.

    Args:
        points (np.arraylike): A `kx3` stack of points in R^3.
        along (np.arraylike): A `(3,)` vector specifying the direction of
            interest.

    Returns:
        int: The index of the most extreme point.
    """
    k = check(locals(), "points", (-1, 3))
    if k == 0:
        raise ValueError("At least one point is required")
    check(locals(), "along", (3,))
    coords_on_axis = points.dot(along)
    return np.argmax(coords_on_axis)


def apex(points, along):
    """
    Find the most extreme point in the direction provided.

    Args:
        points (np.arraylike): A `kx3` stack of points in R^3.
        along (np.arraylike): A `(3,)` vector specifying the direction of
            interest.

    Returns:
        np.ndarray: A copy of a point taken from `points`.
    """
    return points[argapex(points=points, along=along)].copy()


def apex_and_opposite(points, along):
    """
    Find the most extreme point in the direction provided and the most extreme
    point in the opposite direction.

    Args:
        points (np.arraylike): A `kx3` stack of points in R^3.
        along (np.arraylike): A `(3,)` vector specifying the direction of
            interest.

    Returns:
        np.ndarray: A `2x3` vector containing the apex and opposite points.
    """
    k = check(locals(), "points", (-1, 3))
    if k == 0:
        raise ValueError("At least one point is required")
    check(locals(), "along", (3,))
    coords_on_axis = points.dot(along)
    apex = np.argmax(coords_on_axis)
    opposite = np.argmin(coords_on_axis)
    return points[[apex, opposite]].copy()


def nearest(from_points, to_point, ret_index=False):
    """
    Find the point nearest to the given point.

    Args:
        from_points (np.arraylike): A `kx3` stack of points in R^3.
        to_point (np.arraylike): A `(3,)` point of interest.
        ret_index (bool): When `True`, return both the point and its index.

    Returns:
        np.ndarray: A `(3,)` vector taken from `from_points`.
    """
    check(locals(), "from_points", (-1, 3))
    check(locals(), "to_point", (3,))

    absolute_distances = magnitude(from_points - to_point)

    index_of_nearest_point = np.argmin(absolute_distances)
    nearest_point = from_points[index_of_nearest_point]

    if ret_index:
        return nearest_point, index_of_nearest_point
    else:
        return nearest_point


def farthest(from_points, to_point, ret_index=False):
    """
    Find the point farthest from the given point.

    Args:
        from_points (np.arraylike): A `kx3` stack of points in R^3.
        to_point (np.arraylike): A `(3,)` point of interest.
        ret_index (bool): When `True`, return both the point and its index.

    Returns:
        np.ndarray: A `(3,)` vector taken from `from_points`.
    """
    if from_points.ndim != 2 or from_points.shape[1] != 3:
        raise ValueError(
            "Invalid shape %s: farthest expects nx3" % (from_points.shape,)
        )
    if to_point.shape != (3,):
        raise ValueError("to_point should be (3,)")

    absolute_distances = magnitude(from_points - to_point)

    index_of_farthest_point = np.argmax(absolute_distances)
    farthest_point = from_points[index_of_farthest_point]

    if ret_index:
        return farthest_point, index_of_farthest_point
    else:
        return farthest_point


def within(points, radius, of_point, atol=1e-08, ret_indices=False):
    """
    Select points within a given radius of a point.

    Args:
        points (np.arraylike): A `kx3` stack of points in R^3.
        radius (float): The radius of the sphere of interest centered on
            `of_point`.
        of_point (np.arraylike): The `(3,)` point of interest.
        atol (float): The distance tolerance. Points within `radius + atol`
            of `of_point` are selected.
        ret_indexes (bool): When `True`, return both the points and their
            indices.

    Returns:
        np.ndarray: A `(3,)` vector taken from `points`.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Invalid shape %s: within expects nx3" % (points.shape,))
    if not isinstance(radius, float):
        raise ValueError("radius should be a float")
    if of_point.shape != (3,):
        raise ValueError("to_point should be (3,)")

    absolute_distances = magnitude(points - of_point)
    (indices_within_radius,) = (absolute_distances < radius + atol).nonzero()
    points_within_radius = points[indices_within_radius]
    if ret_indices:
        return points_within_radius, indices_within_radius
    else:
        return points_within_radius


def average(values, weights=None, ret_sum_of_weights=False):
    """
    Compute a weighted or unweighted average of the 3D input values. The
    inputs could be points or vectors.

    Args:
        values (np.arraylike): A `kx3` stack of vectors.
        weights (array-convertible): An optional `k` array of weights.
        ret_sum_of_weights (bool): When `True`, the sum of the weights is
            returned. When `weights` is `None`, this is the number of
            elements over which the average is taken.

    Returns:
        np.ndarray: A `(3,)` vector with the weighted or unweighted average.
    """
    k = check(locals(), "values", (-1, 3))
    if weights is not None:
        weights = np.array(weights)
        check(locals(), "weights", (k,))
    result = np.average(values, axis=0, weights=weights)
    if ret_sum_of_weights:
        sum_of_weights = np.sum(weights)
        return result, sum_of_weights
    else:
        return result


def dot(v1, v2):
    """
    Compute individual or pairwise dot products.

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors. If
            stacks are provided for both `v1` and `v2` they must have the
            same shape.
    """
    if v1.ndim == 1 and v2.ndim == 1:
        check(locals(), "v1", (3,))
        check(locals(), "v2", (3,))
        return np.dot(v1, v2)
    else:
        v1, v2 = broadcast_and_tile(v1, v2)
        return np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))


def cross(v1, v2):
    """
    Compute individual or pairwise cross products.

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors. If
            stacks are provided for both `v1` and `v2` they must have the
            same shape.
    """
    if v1.ndim == 1 and v2.ndim == 1:
        check(locals(), "v1", (3,))
        check(locals(), "v2", (3,))
        return np.cross(v1, v2)
    else:
        v1, v2 = broadcast_and_tile(v1, v2)
        return np.cross(v1[:, np.newaxis, :], v2[:, np.newaxis, :])[:, 0, :]


class _BasisVectors(object):
    """
    The cartesian basis vectors.
    """

    @property
    def x(self):
        return np.array([1.0, 0.0, 0.0])

    @property
    def y(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def z(self):
        return np.array([0.0, 0.0, 1.0])

    @property
    def neg_x(self):
        return np.array([-1.0, 0.0, 0.0])

    @property
    def neg_y(self):
        return np.array([0.0, -1.0, 0.0])

    @property
    def neg_z(self):
        return np.array([0.0, 0.0, -1.0])


basis = _BasisVectors()
