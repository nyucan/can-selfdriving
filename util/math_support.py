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


def compute_intercept(slope, x, y):
    """ Compute intercept of the line goes through (x,y) with the given slope.
    """
    return y - slope * x


def slope_function(slope, intercept, x):
    """ This is the tangent line function at point x.
    """
    result = slope * x + intercept
    return result


def compute_points_on_tangent_line(slope, intercept, x_curv, length_tangent_line):
    """ Compute two points on the tangent line that tangent to the mid_fitted line at x_curv.
    """
    x_curv_1 = x_curv - int(length_tangent_line/2)
    y_curv_1 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_1))
    x_curv_2 = x_curv + int(length_tangent_line/2)
    y_curv_2 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_2))
    return x_curv_1, y_curv_1, x_curv_2, y_curv_2


def distance(pt_a, pt_b):
    """ Calculate distance between 2 points.
        @params:
            pt_a: (x, y)
            pt_b: (x, y)
    """
    return math.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)
