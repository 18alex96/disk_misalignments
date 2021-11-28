"""
Functions that support the fitting of the GRAVITY observables
"""

import numpy as np

from inner_disk_models import elliptical_ring

def get_theta0(fit_mode):
    if fit_mode == 'unconstrained':
        theta_0 = np.ma.masked_array(data=[-1.,
                                           0.5,
                                           0.,
                                           0.,
                                           0.,
                                           0.,
                                           np.cos(np.deg2rad(60.)),
                                           np.deg2rad(45.),
                                           0.,
                                           0.],
                                     mask=[False,  # k_c TODO do not leave out this value
                                           False,  # f_c TODO do not leave out this value
                                           False,  # f_h
                                           False,  # f_lor
                                           False,  # l_a
                                           False,  # l_kr
                                           False,  # cosi
                                           False,  # theta
                                           False,  # c_i
                                           False])  # s_i
        indices = {
            'k_c': 0,
            'f_c': 1,
            'f_h': 2,
            'f_lor': 3,
            'l_a': 4,
            'l_kr': 5,
            'cosi': 6,
            'theta': 7,
            'c_i': 8,
            's_i': 9,
        }
    elif fit_mode == 'modulation_amplitude':
        theta_0 = np.ma.masked_array(data=[-1.,
                                           0.5,
                                           0.,
                                           0.,
                                           0.,
                                           0.,
                                           np.cos(np.deg2rad(60.)),
                                           np.deg2rad(45.),
                                           0.,
                                           0.],
                                     mask=[False,  # k_c TODO do not leave out this value
                                           False,  # f_c TODO do not leave out this value
                                           False,  # f_h
                                           False,  # f_lor
                                           False,  # l_a
                                           False,  # l_kr
                                           False,  # cosi
                                           False,  # theta
                                           True,  # c_i
                                           True])  # s_i
        indices = {
            'k_c': 0,
            'f_c': 1,
            'f_h': 2,
            'f_lor': 3,
            'l_a': 4,
            'l_kr': 5,
            'cosi': 6,
            'theta': 7,
            'c_i': None,
            's_i': None,
        }
    elif fit_mode == 'f_lor':
        theta_0 = np.ma.masked_array(data=[-1.,
                                           0.5,
                                           0.,
                                           0.,
                                           0.,
                                           0.,
                                           np.cos(np.deg2rad(60.)),
                                           np.deg2rad(45.),
                                           0.,
                                           0.],
                                     mask=[False,  # k_c TODO do not leave out this value
                                           False,  # f_c TODO do not leave out this value
                                           False,  # f_h
                                           True,  # f_lor
                                           False,  # l_a
                                           False,  # l_kr
                                           False,  # cosi
                                           False,  # theta
                                           False,  # c_i
                                           False])  # s_i
        indices = {
            'k_c': 0,
            'f_c': 1,
            'f_h': 2,
            'f_lor': None,
            'l_a': 3,
            'l_kr': 4,
            'cosi': 5,
            'theta': 6,
            'c_i': 7,
            's_i': 8,
        }
    elif fit_mode == 'f_lor_l_kr':
        theta_0 = np.ma.masked_array(data=[-1.,
                                           0.5,
                                           0.,
                                           0.,
                                           0.,
                                           3.,
                                           np.cos(np.deg2rad(60.)),
                                           np.deg2rad(45.),
                                           0.,
                                           0.],
                                     mask=[False,  # k_c TODO do not leave out this value
                                           False,  # f_c TODO do not leave out this value
                                           False,  # f_h
                                           True,  # f_lor
                                           False,  # l_a
                                           True,  # l_kr
                                           False,  # cosi
                                           False,  # theta
                                           False,  # c_i
                                           False])  # s_i
        indices = {
            'k_c': 0,
            'f_c': 1,
            'f_h': 2,
            'f_lor': None,
            'l_a': 3,
            'l_kr': None,
            'cosi': 4,
            'theta': 5,
            'c_i': 6,
            's_i': 7,
        }

    elif fit_mode == 'f_lor_modulation_amplitude':
        theta_0 = np.ma.masked_array(data=[-1.,
                                           0.5,
                                           0.,
                                           0.,
                                           0.,
                                           0.,
                                           np.cos(np.deg2rad(60.)),
                                           np.deg2rad(45.),
                                           0.,
                                           0.],
                                     mask=[False,  # k_c TODO do not leave out this value
                                           False,  # f_c TODO do not leave out this value
                                           False,  # f_h
                                           True,  # f_lor
                                           False,  # l_a
                                           False,  # l_kr
                                           False,  # cosi
                                           False,  # theta
                                           True,  # c_i
                                           True])  # s_i
        indices = {
            'k_c': 0,
            'f_c': 1,
            'f_h': 2,
            'f_lor': None,
            'l_a': 3,
            'l_kr': 4,
            'cosi': 5,
            'theta': 6,
            'c_i': None,
            's_i': None,
        }

    elif fit_mode == 'l_kr':
        theta_0 = np.ma.masked_array(data=[-1.,
                                           0.5,
                                           0.,
                                           0.,
                                           0.,
                                           3.,
                                           np.cos(np.deg2rad(60.)),
                                           np.deg2rad(45.),
                                           0.,
                                           0.],
                                     mask=[False,  # k_c TODO do not leave out this value
                                           False,  # f_c TODO do not leave out this value
                                           False,  # f_h
                                           False,  # f_lor
                                           False,  # l_a
                                           True,  # l_kr
                                           False,  # cosi
                                           False,  # theta
                                           False,  # c_i
                                           False])  # s_i
        indices = {
            'k_c': 0,
            'f_c': 1,
            'f_h': 2,
            'f_lor': 3,
            'l_a': 4,
            'l_kr': None,
            'cosi': 5,
            'theta': 6,
            'c_i': 7,
            's_i': 8,
        }

    elif fit_mode == 'f_h':
        theta_0 = np.ma.masked_array(data=[-1.,
                                           0.5,
                                           0.,
                                           0.,
                                           0.,
                                           0.,
                                           np.cos(np.deg2rad(60.)),
                                           np.deg2rad(45.),
                                           0.,
                                           0.],
                                     mask=[False,  # k_c TODO do not leave out this value
                                           False,  # f_c TODO do not leave out this value
                                           True,  # f_h
                                           False,  # f_lor
                                           False,  # l_a
                                           False,  # l_kr
                                           False,  # cosi
                                           False,  # theta
                                           False,  # c_i
                                           False])  # s_i
        indices = {
            'k_c': 0,
            'f_c': 1,
            'f_h': None,
            'f_lor': 2,
            'l_a': 3,
            'l_kr': 4,
            'cosi': 5,
            'theta': 6,
            'c_i': 7,
            's_i': 8,
        }
    else:
        raise ValueError(f'Fit mode {fit_mode} not supported')

    return theta_0, indices


def func_cps(theta, masked_params, x_vis2, x_cp):
    theta_input = np.zeros(len(masked_params))
    count = 0
    for i, tmp_mask in enumerate(masked_params.mask):
        if not tmp_mask:
            theta_input[i] = theta[count]
            count += 1
        else:
            theta_input[i] = masked_params.data[i]

    tmp_model = elliptical_ring(*theta_input)
    tmp_vis2 = np.abs(tmp_model.get_full_visibility(*x_vis2)) ** 2
    tmp_cp = tmp_model.get_closure_phase(*x_cp)

    return np.hstack([tmp_vis2, tmp_cp])


def log_like(theta, func, y, y_err, masked_params, x_vis2, x_cp, f_d, e_f_d):
    sigma2 = y_err ** 2

    # add additional constraint that f_c + f_h < 1
    if theta.data[1] + theta.data[2] > 1:
        return -np.inf

    # difference between measured and model value
    delta_y = y - func(theta, masked_params, x_vis2, x_cp)

    # return log likelihood
    # the second term refers to the additional constrian from L17, page 9
    return -.5 * (np.sum(delta_y ** 2 / sigma2 + np.log(2 * np.pi * sigma2)) + (theta[1] - f_d) ** 2 / e_f_d ** 2)


def log_prior(theta, theta_ranges):
    for i, tmp_theta in enumerate(theta):
        if tmp_theta < theta_ranges[i][0] or tmp_theta > theta_ranges[i][1]:
            return -np.inf
    return 0.


def log_prob(theta, func, y, y_err, theta_ranges, masked_params, x_vis2, x_cp, f_d, e_f_d):
    lp = log_prior(theta, theta_ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(theta=theta,
                         func=func,
                         y=y,
                         y_err=y_err,
                         masked_params=masked_params,
                         x_vis2=x_vis2,
                         x_cp=x_cp,
                         f_d=f_d,
                         e_f_d=e_f_d)