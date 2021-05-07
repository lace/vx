import numpy as np
import pytest
from . import core as vg


def test_apex_functions():
    points = np.array(
        [
            [-0.97418884, -0.79808404, -0.18545491],
            [0.60675227, 0.32673201, -0.20369793],
            [0.67040405, 0.19267665, -0.56983579],
            [-0.68038753, -0.90011588, 0.4649872],
            [-0.62813991, -0.23947753, 0.07933854],
            [0.26348356, 0.23701114, -0.38230596],
            [0.08302473, 0.2784907, 0.09308946],
            [0.58695587, -0.33253376, -0.33493078],
            [-0.39221704, -0.45240036, 0.25284163],
            [0.46270635, -0.3865265, -0.98106526],
        ]
    )

    np.testing.assert_array_equal(vg.argapex(points, along=vg.basis.x), 2)
    np.testing.assert_array_equal(vg.apex(points, along=vg.basis.x), points[2])
    np.testing.assert_array_equal(
        vg.apex_and_opposite(points, along=vg.basis.x), points[[2, 0]]
    )
    np.testing.assert_array_equal(vg.argapex(points, along=vg.basis.neg_x), 0)
    np.testing.assert_array_equal(vg.apex(points, along=vg.basis.neg_x), points[0])
    np.testing.assert_array_equal(
        vg.apex_and_opposite(points, along=vg.basis.neg_x), points[[0, 2]]
    )
    np.testing.assert_array_equal(vg.argapex(points, along=vg.basis.y), 1)
    np.testing.assert_array_equal(vg.apex(points, along=vg.basis.y), points[1])
    np.testing.assert_array_equal(
        vg.apex_and_opposite(points, along=vg.basis.y), points[[1, 3]]
    )
    np.testing.assert_array_equal(vg.argapex(points, along=vg.basis.neg_y), 3)
    np.testing.assert_array_equal(vg.apex(points, along=vg.basis.neg_y), points[3])
    np.testing.assert_array_equal(
        vg.apex_and_opposite(points, along=vg.basis.neg_y), points[[3, 1]]
    )
    np.testing.assert_array_equal(vg.argapex(points, along=vg.basis.z), 3)
    np.testing.assert_array_equal(vg.apex(points, along=vg.basis.z), points[3])
    np.testing.assert_array_equal(
        vg.apex_and_opposite(points, along=vg.basis.z), points[[3, 9]]
    )
    np.testing.assert_array_equal(vg.argapex(points, along=vg.basis.neg_z), 9)
    np.testing.assert_array_equal(vg.apex(points, along=vg.basis.neg_z), points[9])
    np.testing.assert_array_equal(
        vg.apex_and_opposite(points, along=vg.basis.neg_z), points[[9, 3]]
    )

    v = np.full(3, 1 / 3 ** 0.5)
    expected = points[np.argmax(points.sum(axis=1))]
    np.testing.assert_array_equal(vg.apex(points, along=v), expected)

    # Test non-normalized too.
    np.testing.assert_array_equal(vg.apex(points, along=np.array([1, 1, 1])), expected)

    with pytest.raises(ValueError, match=r"At least one point is required"):
        vg.apex(np.zeros((0, 3)), vg.basis.z)
    with pytest.raises(ValueError, match=r"At least one point is required"):
        vg.apex_and_opposite(np.zeros((0, 3)), vg.basis.z)


def test_apex_returns_a_copy():
    points = np.array(
        [
            [-0.97418884, -0.79808404, -0.18545491],
            [0.60675227, 0.32673201, -0.20369793],
            [0.67040405, 0.19267665, -0.56983579],
            [-0.68038753, -0.90011588, 0.4649872],
            [-0.62813991, -0.23947753, 0.07933854],
            [0.26348356, 0.23701114, -0.38230596],
            [0.08302473, 0.2784907, 0.09308946],
            [0.58695587, -0.33253376, -0.33493078],
            [-0.39221704, -0.45240036, 0.25284163],
            [0.46270635, -0.3865265, -0.98106526],
        ]
    )
    result = vg.apex(points, along=vg.basis.x)
    result[1] = 5.0
    np.testing.assert_array_equal(points[2], [0.67040405, 0.19267665, -0.56983579])


def test_apex_and_opposite_returns_a_copy():
    points = np.array(
        [
            [-0.97418884, -0.79808404, -0.18545491],
            [0.60675227, 0.32673201, -0.20369793],
            [0.67040405, 0.19267665, -0.56983579],
            [-0.68038753, -0.90011588, 0.4649872],
            [-0.62813991, -0.23947753, 0.07933854],
            [0.26348356, 0.23701114, -0.38230596],
            [0.08302473, 0.2784907, 0.09308946],
            [0.58695587, -0.33253376, -0.33493078],
            [-0.39221704, -0.45240036, 0.25284163],
            [0.46270635, -0.3865265, -0.98106526],
        ]
    )
    result = vg.apex_and_opposite(points, along=vg.basis.x)
    # Update all the y coordinates.
    result[:, 1] = 5.0
    np.testing.assert_array_equal(points[0], [-0.97418884, -0.79808404, -0.18545491])
    np.testing.assert_array_equal(points[2], [0.67040405, 0.19267665, -0.56983579])
