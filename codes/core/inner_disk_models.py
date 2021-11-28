import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.special import j0, jn
from matplotlib import pyplot as plt
from functions import hybrid_gauss_lorentz_ft_rad
import os
import matplotlib as mpl

class elliptical_ring():
    """
    Defines the parametric model for our analysis
    """

    def __init__(self,
                 k_c,
                 f_c,
                 f_h,
                 f_lor,
                 l_a,
                 l_kr,
                 cos_i,
                 theta,
                 c_j,
                 s_j):
        # set up object parameters
        self.k_c = k_c
        self.f_c = f_c
        self.f_h = f_h
        self.f_lor = f_lor
        self.l_a = l_a
        self.l_kr = l_kr
        self.cos_i = cos_i
        self.theta = theta
        self.c_j = c_j
        self.s_j = s_j

        # derived parameters
        self.a_r = 10 ** self.l_a / np.sqrt(1 + 10 ** (2 * self.l_kr))
        self.a_k = 10 ** self.l_kr * self.a_r

        self.rho_j = np.sqrt(self.c_j ** 2 + self.s_j ** 2)
        self.theta_j = np.arctan2(self.s_j, -self.c_j) + np.pi / 2

        self.f_s = 1. - self.f_h - self.f_c

    def get_parameters(self):
        return np.array([self.k_c, self.f_c, self.f_h, self.f_lor, self.l_a, self.l_kr, self.cos_i, self.theta, self.c_j, self.s_j])

    def get_disk_visibility(self,
                            u,
                            v,
                            lam):

        q = np.sqrt(u ** 2 + v ** 2) / lam * np.pi / 180. / 3600 / 1000.    # convert to 1 / mas
        psi = np.arctan2(v, -u) + np.pi

        # plt.imshow((psi),
        #            origin='lower',
        #            extent=(np.min(v), np.max(v), np.min(u), np.max(u)))
        # plt.show()

        # calculate azimuthal modulation
        sum = 0.
        if not hasattr(self.c_j, "__len__"):
            sum += (-1j) ** 1 * self.rho_j * np.cos(1 * (psi - self.theta - self.theta_j)) * \
                   jn(2 * np.pi * q * self.a_r * np.sqrt((np.cos(psi - self.theta) * self.cos_i) ** 2 +
                                                        np.sin(psi - self.theta) ** 2), 1)
        else:
            for i in range(1, len(self.rho_j) + 1):
                sum += (-1j) ** i * self.rho_j[i - 1] * np.cos(i * (psi - self.theta - self.theta_j[i - 1])) * \
                       jn(2 * np.pi * q * self.a_r * np.sqrt((np.cos(psi - self.theta) * self.cos_i) ** 2 +
                                                             np.sin(psi - self.theta) ** 2), i)

        # get preliminary visibility
        tmp_vis = j0(2 * np.pi * q * self.a_r * np.sqrt((np.cos(psi - self.theta) * self.cos_i) ** 2 +
                                                        np.sin(psi - self.theta) ** 2)) + sum

        hybrid_gauss_lorentz = hybrid_gauss_lorentz_ft_rad(q, psi, [self.f_lor, self.a_k, self.cos_i, self.theta])

        # convolve preliminray visibility with a hybrid Gaussian Lorentzian kernel
        disk_vis = tmp_vis * hybrid_gauss_lorentz

        return disk_vis

    def get_full_visibility(self,
                            u,
                            v,
                            lam_0,
                            lam,
                            k_s):

        disk_vis = self.get_disk_visibility(u, v, lam)

        full_vis = (self.f_s * (lam_0 / lam) ** k_s + disk_vis * self.f_c * (lam_0 / lam) ** self.k_c) / \
                   ((self.f_s + self.f_h) * (lam_0 / lam) ** k_s + self.f_c * (lam_0 / lam) ** self.k_c)

        return full_vis

    def get_closure_phase(self,
                          u_coord_1,
                          u_coord_2,
                          v_coord_1,
                          v_coord_2,
                          lam_0,
                          lam,
                          k_s):

        # either derive individual phases and add the, for CPs (1) or derive CPs via triple product (2)
        # both should be identical
        full_vis_1_1 = self.get_full_visibility(u_coord_1, v_coord_1, lam_0, lam, k_s)
        full_vis_1_2 = self.get_full_visibility(u_coord_2, v_coord_2, lam_0, lam, k_s)
        full_vis_1_3 = self.get_full_visibility(-(u_coord_1 + u_coord_2), -(v_coord_1 + v_coord_2), lam_0, lam, k_s)
        full_vis_2_1 = self.get_full_visibility(u_coord_1, v_coord_1, lam_0, lam, k_s)
        full_vis_2_2 = self.get_full_visibility(u_coord_2, v_coord_2, lam_0, lam, k_s)
        full_vis_2_3 = np.conjugate(self.get_full_visibility(u_coord_1 + u_coord_2, v_coord_1 + v_coord_2, lam_0, lam, k_s))

        phase_1_1 = np.arctan2(np.imag(full_vis_1_1), np.real(full_vis_1_1))
        phase_1_2 = np.arctan2(np.imag(full_vis_1_2), np.real(full_vis_1_2))
        phase_1_3 = np.arctan2(np.imag(full_vis_1_3), np.real(full_vis_1_3))

        closure_phase_1 = phase_1_1 + phase_1_2 + phase_1_3

        triple_prod = full_vis_2_1 * full_vis_2_2 * full_vis_2_3
        closure_phase_2 = np.arctan2(np.imag(triple_prod), np.real(triple_prod))

        mask_cp_1_1 = closure_phase_1 > np.pi
        mask_cp_1_2 = closure_phase_1 <= -np.pi
        mask_cp_2_1 = closure_phase_2 > np.pi
        mask_cp_2_2 = closure_phase_2 <= -np.pi

        closure_phase_1[mask_cp_1_1] = closure_phase_1[mask_cp_1_1] - 2 * np.pi
        closure_phase_1[mask_cp_1_2] = closure_phase_1[mask_cp_1_2] + 2 * np.pi
        closure_phase_2[mask_cp_2_1] = closure_phase_2[mask_cp_2_1] - 2 * np.pi
        closure_phase_2[mask_cp_2_2] = closure_phase_2[mask_cp_2_2] + 2 * np.pi

        return np.rad2deg(closure_phase_2)


    def plot_model(self,
                   u,
                   v,
                   lam_0,
                   lam,
                   k_s,
                   mode='vis2',
                   path_save=None,
                   return_fig=False):

        full_vis = self.get_full_visibility(u, v, lam_0, lam, k_s)

        pixscale = lam / (2 * np.max(u)) / np.pi * 180 * 3600 * 1000
        im_size = u.shape[0]

        # artificially enhance sampling of uv plane
        sampling = 1 / (10 ** (self.l_a-2))
        vv, uu = np.mgrid[sampling*np.min(u):sampling*np.max(u)+sampling:sampling, sampling*np.min(v):sampling*np.max(v)+sampling:sampling]

        disk_vis = self.get_disk_visibility(uu, vv, lam_0)

        f, (ax_1, ax_2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

        if mode == 'vis2':
            cbar_norm = mpl.colors.Normalize(vmin=0., vmax=1.)
            im_plot = np.abs(full_vis) ** 2
        elif mode == 'real':
            cbar_norm = mpl.colors.Normalize(vmin=-1., vmax=1.)
            im_plot = np.real(full_vis)
        elif mode == 'phase':
            cbar_norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
            im_plot = np.arctan2(np.imag(disk_vis), np.real(disk_vis))
        else:
            raise ValueError(f'Mode {mode} not supported.')

        ax_1.imshow(im_plot,
                    origin='lower',
                    norm=cbar_norm,
                    extent=(np.min(v), np.max(v), np.min(u), np.max(u)))

        ax_1.set_xlabel('u [m]')
        ax_1.set_ylabel('v [m]')

        ax_2.imshow(np.sqrt(np.abs(np.fft.ifftshift(np.fft.ifft2(disk_vis))) ** 2),
                    origin='lower',
                    extent=(-(im_size - 1) / 2 * pixscale / sampling,
                            (im_size - 1) / 2 * pixscale / sampling,
                            -(im_size - 1) / 2 * pixscale / sampling,
                            (im_size - 1) / 2 * pixscale / sampling ))
        ax_2.plot([], [], linestyle='none',
                  label=f'k_c = {self.k_c:.2f}\nf_c = {self.f_c:.2f}\nf_h = {self.f_h:.2f}\nf_lor = {self.f_lor:.2f}\nl_a = {self.l_a:.2f}\nl_kr = {self.l_kr:.2f}\ni = {np.rad2deg(np.arccos(self.cos_i)):.1f}\ntheta = {np.rad2deg(self.theta):.1f}\nc_j = {self.c_j:.2f}\ns_j = {self.s_j:.2f}')

        ax_2.invert_xaxis()
        ax_2.set_xlabel(r'$\Delta$RA [mas]')
        ax_2.set_ylabel(r'$\Delta$Dec [mas]')
        ax_2.legend(loc=0)

        if return_fig:
            return f, ax_1, ax_2

        # save results
        if path_save is not None:
            f.savefig(path_save,
                      bbox_inches='tight',
                      transparent=True,
                      pad_inches=0)
            plt.close(f)
        else:
            plt.show()

        # save results
        if path_save is not None:
            f.savefig(path_save,
                      bbox_inches='tight',
                      transparent=True,
                      pad_inches=0)
            plt.close(f)
        else:
            plt.show()

if __name__ == '__main__':

    path_save = '/Users/alex/surfdrive/PhD/Research/Data/surveys/gravity_inner_disks/inner_disk_models/testing/'

    V, U = np.mgrid[-150:151, -150:151]
    lam = 2.25e-6

    # LKCA 15 test
    k_c = -3
    f_c = .7
    f_h = 0.1
    f_lor = 0.
    l_a = -0.7
    l_kr = 0.
    cos_i = np.cos(np.deg2rad(55))
    theta = np.deg2rad(60)
    c_j = 0.
    s_j = 0.
    # k_c = -5.75
    # f_c = .73
    # f_h = 0.03
    # f_lor = 0.31
    # l_a = 0.37
    # l_kr = 0.08
    # cos_i = 0.91 # np.cos(np.deg2rad(0))
    # theta = 1.4 # np.deg2rad(0)
    # c_j = 0.97
    # s_j = 0.18
    # k_c = -4.12
    # f_c = .55
    # f_h = 0.03
    # f_lor = 1.
    # l_a = 0.98
    # l_kr = -0.26
    # cos_i = 0.63 # np.cos(np.deg2rad(0))
    # theta = 1.2 # np.deg2rad(0)
    # c_j = -0.18
    # s_j = 0.98
    # k_c = -3.25
    # f_c = .63
    # f_h = 0.12
    # f_lor = 1.
    # l_a = 0.33
    # l_kr = -0.08
    # cos_i = 0.53 # np.cos(np.deg2rad(0))
    # theta = 1.26 # np.deg2rad(0)
    # c_j = 1.
    # s_j = -0.02
    # k_c = -5.31
    # f_c = .30
    # f_h = 0.09
    # f_lor = 0.83
    # l_a = 0.42
    # l_kr = -0.21
    # cos_i = 0.66 # np.cos(np.deg2rad(0))
    # theta = 1.41 # np.deg2rad(0)
    # c_j = -0.01
    # s_j = 0.03
    # k_c = -2.15
    # f_c = .58
    # f_h = 0.0
    # f_lor = 0.
    # l_a = 0.27
    # l_kr = -0.26
    # cos_i = 0.86 # np.cos(np.deg2rad(0))
    # theta = 2.78 # np.deg2rad(0)
    # c_j = -0.16
    # s_j = 0.10
    # k_c = -2.48
    # f_c = .6
    # f_h = 0.0
    # f_lor = 0.21
    # l_a = 0.30
    # l_kr = 0.2
    # cos_i = 0.61 # np.cos(np.deg2rad(0))
    # theta = 2.22 # np.deg2rad(0)
    # c_j = -0.39
    # s_j = 0.92
    # k_c = -2
    # f_c = .6
    # f_h = 0.1
    # f_lor = 0.5
    # l_a = 0.3
    # l_kr = -0.5
    # cos_i = np.cos(np.deg2rad(60))
    # theta = np.deg2rad(30)
    # c_j = 0.
    # s_j = -1.
    # # HD 38120 (Perraut19)
    # k_c = -1.5729294717346385
    # f_c = .64
    # f_h = 0.0
    # f_lor = 0.48
    # fwhm = 6.47
    # a = fwhm / 2
    # w = 0.74
    # a_k = w * a
    # a_r = np.sqrt(a ** 2 - a_k ** 2)
    # l_a = np.log10(a)
    # l_kr = np.log10(a_k / a_r)
    # cos_i = 0.66
    # theta = np.deg2rad(165)
    # c_j = 0.
    # s_j = 0.
    # # HD 97048 (Perraut19)
    # k_c = -2.1990363338696293
    # f_c = .73
    # f_h = 0.0
    # f_lor = 1.
    # fwhm = 4.38
    # a = fwhm / 2
    # w = 0.83
    # a_k = w * a
    # a_r = np.sqrt(a ** 2 - a_k ** 2)
    # l_a = np.log10(a)
    # l_kr = np.log10(a_k / a_r)
    # cos_i = 0.66
    # theta = np.deg2rad(179)
    # c_j = 0.
    # s_j = 0.
    # # HD 100546 (Perraut19)
    # k_c = -1.9585082765939588
    # f_c = .63
    # f_h = 0.0
    # f_lor = 0.99
    # fwhm = 5.02
    # a = fwhm / 2
    # w = 0.46
    # a_k = w * a
    # a_r = np.sqrt(a ** 2 - a_k ** 2)
    # l_a = np.log10(a)
    # l_kr = np.log10(a_k / a_r)
    # cos_i = 0.64
    # theta = np.deg2rad(146)
    # c_j = 0.
    # s_j = 0.
    # # HD 135244B (Perraut19)
    # k_c = -1.1997887172548003
    # f_c = .56
    # f_h = 0.0
    # f_lor = 0.0
    # fwhm = 2.96
    # a = fwhm / 2
    # w = 0.38
    # a_k = w * a
    # a_r = np.sqrt(a ** 2 - a_k ** 2)
    # l_a = np.log10(a)
    # l_kr = np.log10(a_k / a_r)
    # cos_i = 0.76
    # theta = np.deg2rad(30)
    # c_j = 0.
    # s_j = 0.
    # k_c = -1
    # f_c = .9
    # f_h = 0.0
    # f_lor = 0.
    # l_a = 0.3
    # l_kr = -.3
    # cos_i = np.cos(np.deg2rad(50))
    # theta = np.deg2rad(40)
    # c_j = 0.
    # s_j = 0.


    model = elliptical_ring(k_c=k_c,
                            f_c=f_c,
                            f_h=f_h,
                            f_lor=f_lor,
                            l_a=l_a,
                            l_kr=l_kr,
                            cos_i=cos_i,
                            theta=theta,
                            c_j=c_j,
                            s_j=s_j)

    disk_vis = model.get_disk_visibility(U, V, lam)
    full_vis = model.get_full_visibility(U, V, lam, lam, 1.)

    u_1 = np.random.rand(100) * 300 - 150
    u_2 = np.random.rand(100) * 300 - 150
    u_3 = - (u_1 + u_2)
    v_1 = np.random.rand(100) * 300 - 150
    v_2 = np.random.rand(100) * 300 - 150
    v_3 = - (v_1 + v_2)

    # u_coord_cps = np.array([[l, m, n] for (l, m, n) in zip(u_1, u_2, u_3)])
    # v_coord_cps = np.array([[l, m, n] for (l, m, n) in zip(v_1, v_2, v_3)])

    # model.get_closure_phase(np.array([u_1, u_2, u_3]), np.array([v_1, v_2, v_3]), lam, lam, 1.)
    print(model.get_closure_phase(u_1, u_2, v_1, v_2, lam, lam, 1.))

    model.plot_model(u=U,
                     v=V,
                     lam_0=lam,
                     lam=lam,
                     k_s=1.,
                     mode='vis2',
                     path_save=None)
