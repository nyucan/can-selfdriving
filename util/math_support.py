# python 2.7
from __future__ import absolute_import, division, print_function
import math

def first_order_derivative(w, x, order=2):
    """ Return the value of dy/dx at point x.
        y = w[0] * x^2 + w[1] * x + w[2]
    """
    result = 2 * w[0] * x + w[1]
    return result


def second_order_derivative(w, x, order=2):
    """ Return the value of d2y/dx2 at point x.
    """
    result = 2 * w[0]
    return result


def curvature(w, x, order=2):
    fd = first_order_derivative(w, x, order)
    sd = second_order_derivative(w, x, order)
    return sd / (1 + fd**2) ** 1.5


def radian(distance_at_mid, image_center):
    tangent = distance_at_mid / image_center
    if (tangent > 0):
        return math.atan(tangent)
    else:
        return -math.atan(-tangent)

