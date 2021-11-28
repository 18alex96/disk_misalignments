import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt



deg2rad = np.pi / 180.
rad2deg = 1. / deg2rad

def get_mis(inc_in=None, pa_in=None, inc_out=None, pa_out=None):
    """
    calculates the misalignment angle between the inner and outer disk [radian]

    Parameters
    ----------
    inc_in          : float
                      Inclination angle of the inner disk [radian]

    pa_in           : float
                      Position angle of the inner disk [radian]

    inc_out         : float
                      Inclination angle of the outer disk [radian]

    pa_out          : float
                      Position angle of the outer disk [radian]
    """
    cos_mis = np.sin(inc_in) * np.sin(inc_out) * np.cos(pa_in - pa_out) + np.cos(inc_in) * np.cos(inc_out)
    mis = np.arccos(cos_mis)

    return mis

def get_shadow_pa(inc_in=None, pa_in=None, inc_out=None, pa_out=None):
    """
    calculates the position angleof the shadows

    Parameters
    ----------
    inc_in          : float
                      Inclination angle of the inner disk [radian]

    pa_in           : float
                      Position angle of the inner disk [radian]

    inc_out         : float
                      Inclination angle of the outer disk [radian]

    pa_out          : float
                      Position angle of the outer disk [radian]
    """
    ax = np.sin(inc_in) * np.cos(inc_out) * np.cos(pa_in) - np.cos(inc_in) * np.sin(inc_out) * np.cos(pa_out)
    ay = np.sin(inc_in) * np.cos(inc_out) * np.sin(pa_in) - np.cos(inc_in) * np.sin(inc_out) * np.sin(pa_out)
    return np.arctan2(ay, ax) + np.pi

def get_shadow_x(inc_in=None, pa_in=None, inc_out=None, pa_out=None, h=None):
    """
    calculates the position angleof the shadows

    Parameters
    ----------
    inc_in          : float
                      Inclination angle of the inner disk [radian]

    pa_in           : float
                      Position angle of the inner disk [radian]

    inc_out         : float
                      Inclination angle of the outer disk [radian]

    pa_out          : float
                      Position angle of the outer disk [radian]
    h               : float
                      Height of the scattering surface [au]
    """
    x = h * np.cos(inc_in) / (np.cos(inc_out) * np.sin(inc_in) * np.sin(pa_in) - np.cos(inc_in) * np.sin(inc_out) * np.sin(pa_out))
    return x

if __name__ == '__main__':
    # # VALUES LP
    # ii = 47.9 * deg2rad
    # pai = 81 * deg2rad
    # io = -38 * deg2rad
    # pao = 142 * deg2rad
    # h = 4.2
    # ii = 46 * deg2rad
    # pai = (82+0) * deg2rad
    # io = 34 * deg2rad
    # pao = (324+0) * deg2rad
    # h = 4

    ii = 24 * deg2rad
    pai = (15+0) * deg2rad
    io = 38 * deg2rad
    pao = (163+0) * deg2rad
    h = 40
    #
    #
    # mis = get_mis(inc_in=ii, pa_in=pai, inc_out=io, pa_out=pao) * rad2deg
    #
    # print(mis)
    #
    alpha = get_shadow_pa(inc_in=ii, pa_in=pai, inc_out=io, pa_out=pao) * rad2deg

    print(alpha)

    x = get_shadow_x(inc_in=ii, pa_in=pai, inc_out=io, pa_out=pao, h=h)

    print(x)

    # # test this framework for our measurements
    # path_fits = '/Users/alex/surfdrive/PhD/Research/Data/surveys/gravity_inner_disks/data/sphere_scattered_light/HD_100453/HD100453_ZIMPOL_Ip_Qphi.fits'
    # pixscale = 0.0068
    # dist = 1000 / 9.635549075319966
    #
    # i_gravity = np.deg2rad(46.05)
    # pa_gravity = np.deg2rad(81.58)
    # i_alma = np.deg2rad(33.81)
    # pa_alma = np.deg2rad(324.35)
    # h_est = 4
    #
    # mis = get_mis(inc_in=i_gravity, pa_in=pa_gravity, inc_out=i_alma, pa_out=pa_alma) * rad2deg
    #
    # print(mis)
    #
    # alpha = get_shadow_pa(inc_in=i_gravity, pa_in=pa_gravity, inc_out=i_alma, pa_out=pa_alma) * rad2deg
    #
    # print(alpha)
    #
    # x = get_shadow_x(inc_in=i_gravity, pa_in=pa_gravity, inc_out=i_alma, pa_out=pa_alma, h=h_est)
    #
    # print(x)
    #
    # m = np.tan(np.deg2rad(alpha) - np.pi / 2)
    # print(m)
    #
    # hdu = fits.open(path_fits)
    # im = hdu[0].data
    # im_size = im.shape[0]
    #
    # xx = np.linspace(-(im_size - 1) / 2 * pixscale, (im_size - 1) / 2 * pixscale, 1000)
    # yy = m * xx + x / dist
    #
    # plt.imshow(np.log10(im),
    #            origin='lower',
    #            extent=(-(im_size - 1) / 2 * pixscale,
    #                    (im_size - 1) / 2 * pixscale,
    #                    -(im_size - 1) / 2 * pixscale,
    #                    (im_size - 1) / 2 * pixscale))
    #
    # plt.plot(xx, yy)
    #
    # plt.show()