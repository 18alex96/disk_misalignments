"""
Useful functions to describe inner disk models
"""
import numpy as np

def gauss_2d_rad(r, phi, params):
    a, cos_i, theta = params

    return np.log(2) / (np.pi * a ** 2 * cos_i) * np.exp(
        -np.log(2) * ((r * np.cos(phi - theta) / a) ** 2 + (r * np.sin(phi - theta) / (a * cos_i)) ** 2))


def lorentz_2d_rad(r, phi, params):
    a, cos_i, theta = params

    tmp_f = a / (2 * np.pi * np.sqrt(3)) * (
            a ** 2 / 3 + (r * np.cos(phi - theta)) ** 2 + ((r * np.sin(phi - theta) / cos_i) ** 2)) ** (-3 / 2)

    return tmp_f


def hybrid_gauss_lorentz_rad(r, phi, params):
    f_lor, a, cos_i, theta = params

    return (1 - f_lor) * gauss_2d_rad(r, phi, [*params[1:]]) + f_lor * lorentz_2d_rad(r, phi, [*params[1:]])


def gauss_2d_ft_rad(q, psi, params):
    a, cos_i, theta = params

    tmp_f = np.exp(-1 * (np.pi * q * a) ** 2 * ((np.cos(psi - theta) * cos_i) ** 2 +
                                                (np.sin(psi - theta) ** 2)) / np.log(2))

    return tmp_f



def lorentz_2d_ft_rad(q, psi, params):
    a, cos_i, theta = params

    tmp_f = np.exp(- 2 * np.pi / np.sqrt(3) * q * a * np.sqrt((np.cos(psi - theta) * cos_i) ** 2 +
                                                              (np.sin(psi - theta)) ** 2))
    #
    # tmp_f = a / (2 * np.pi * np.sqrt(3)) * (
    #         a ** 2 / 3 + (r * np.cos(phi - theta)) ** 2 + ((r * np.sin(phi - theta) / cos_i) ** 2)) ** (-3 / 2)

    return tmp_f

def hybrid_gauss_lorentz_ft_rad(q, psi, params):
    f_lor, a, cos_i, theta = params

    return (1 - f_lor) * gauss_2d_ft_rad(q, psi, [*params[1:]]) + f_lor * lorentz_2d_ft_rad(q, psi, [*params[1:]])