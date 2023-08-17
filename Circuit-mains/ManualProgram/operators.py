import math


def c_equal(n1):
    return n1


def c_double(n1):
    return n1 * 2


def c_half(n1):
    return n1 / 2


def c_add(n1, n2):
    return n1 + n2


def c_minus(n1, n2):
    return math.fabs(n1 - n2)


def c_mul(n1, n2):
    return n1 * n2


def c_three_mul(n1, n2, n3):
    return n1 * n2 * n3


def c_divide(n1, n2):
    if n1 > 0 and n2 > 0:
        return n1 / n2
    return False


def c_four_mul(n1, n2, n3, n4):
    return n1 * n2 * n3 * n4
