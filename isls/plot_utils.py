import numpy as np
import scipy
from scipy.linalg import block_diag, null_space
import scipy.sparse as sp
from scipy.special import factorial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def plot_robot(xs, color='k', xlim=None, ax=None, ylim=None, robot_base=False,**kwargs):
    if not ax:
        l = plt.plot(xs[:, 0], xs[:, 1], marker='o', color=color, lw=10, mfc='w', solid_capstyle='round',
                     **kwargs)

        plt.gca().set_aspect('equal')

        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
    else:
        l = ax.plot(xs[:, 0], xs[:, 1], color=color,marker='o', mec="k", mfc="w", lw=10, solid_capstyle='round',
                    **kwargs)
        if robot_base:
            plot_robot_base(xs[0], ax, ec="k", fc="k", sz=0.1, alpha=0.8, zorder=1)

        ax.set_aspect('equal')

        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)



    return l

def plot_robot_base(p1, ax,ec="k",fc="blue", sz=1.2, alpha=1., **kwargs):
    nbSegm = 30
    sz = sz * 1.2
    t1 = np.linspace(0, np.pi, nbSegm - 2)
    xTmp = np.zeros((2, nbSegm)) # 2D only
    xTmp[0, 0] = sz * 1.5
    xTmp[0, 1:-1] = sz * 1.5 * np.cos(t1)
    xTmp[0, -1] = -sz * 1.5

    xTmp[1, 0] = -sz * 1.2
    xTmp[1, 1:-1] = sz * 1.5 * np.sin(t1)
    xTmp[1, -1] = -sz * 1.2

    x = xTmp + np.tile(p1[:,None], (1, nbSegm))
    patch = mpatches.Polygon(x.T[:,:2], ec=ec, fc=fc, alpha=alpha, lw=3, **kwargs)
    ax.add_patch(patch)

    nb_line = 4
    mult = 1.2
    xTmp2 = np.zeros((2, nb_line))  # 2D only
    xTmp2[0,:] = np.linspace(-sz * mult, sz * mult, nb_line)
    xTmp2[1,:] = [-sz * mult] * nb_line

    x2 = xTmp2 + np.tile((p1 + np.array([0.04,0.05]))[:,None], (1, nb_line))
    x3 = xTmp2 + np.tile((p1 + np.array([-0.5, -1])*sz)[:,None], (1, nb_line))

    for i in range(nb_line):
        tmp = np.zeros((2, 2)) # N*2
        tmp[0] = [x2[0,i], x2[1,i]]
        tmp[1] = [x3[0, i], x3[1, i]]
        patch = Line2D(tmp[:,0], tmp[:, 1],  color=ec, alpha=alpha, lw=2)
        ax.add_line(patch)


def rrect(wlc, color, ax=None):
    # draw a rounded rectangle (using complex numbers and a kronecker sum :-)
    N = 25  # number of points per corner
    width = wlc[0]
    length = wlc[1]
    curve = wlc[2]
    a = np.linspace(0, 2 * np.pi, 4 * N)
    circle = curve * np.exp(1j * a)
    width = width - curve
    length = length - curve
    rect1 = np.diag(width * np.array([1, -1, -1, 1]) + 1j * length * np.array([1, 1, -1, -1]))
    rectN = np.sum(np.kron(rect1, np.ones((1, N))), axis=0)
    rr = circle + rectN
    rr = np.append(rr, rr[0])  # close the curve
    h = mpatches.Polygon(np.array([np.real(rr), np.imag(rr)]).T, color=color)
    return h


def twist(obj, x, y, theta=0):
    # a planar twist: rotate object by theta, then translate by (x,y)
    for h in obj:
        xy = h.get_xy()
        Z = xy[:, 0] + 1j * xy[:, 1]
        Z = Z * np.exp(1j * theta)
        Z = Z + (x + 1j * y)
        h.set_xy(np.stack([np.real(Z), np.imag(Z)]).T)


def plot_car(x, u):
    body = np.array([0.9, 2.1, 0.3])  # body = [width length curvature]
    bodycolor = 0.7 * np.array([1, 1, 1])
    headlights = np.array([0.25, 0.1, .1, body[0] / 2])  # headlights [width length curvature x]
    lightcolor = np.array([1, 1, 0])
    wheel = np.array([0.15, 0.4, .06, 1.1 * body[0], -1.1, .9])  # wheels = [width length curvature x yb yf]
    wheelcolor = 'k'
    h = []
    # make wheels
    for front in range(2):
        for right in [-1, 1]:
            h += [rrect(wheel, wheelcolor)]  ##ok<AGROW>
            if front == 1:
                twist([h[-1]], 0, 0, u[0])
            twist([h[-1]], right * wheel[3], wheel[3 + front])
    # make body
    h += [rrect(body, bodycolor)]

    # make window (hard coded)
    h += [mpatches.Polygon(np.stack([[-.8, .8, .7, -.7], .6 + .3 * np.array([1, 1, -1, -1])]).T, color='w')]

    # headlights
    h += [rrect(headlights[0:3], lightcolor)]
    twist([h[-1]], headlights[3], body[1] - headlights[1])
    h += [rrect(headlights[0:3], lightcolor)]
    twist([h[-1]], -headlights[3], body[1] - headlights[1])
    # put rear wheels at (0,0)
    twist(h, 0, -wheel[4])
    # align to x-axis
    twist(h, 0, 0, -np.pi / 2)
    # make origin (hard coded)
    ol = 0.1
    ow = 0.01
    h += [mpatches.Polygon(np.stack([ol * np.array([-1, 1, 1, -1]), ow * np.array([1, 1, -1, -1])]).T, color='k')]
    h += [mpatches.Polygon(np.stack([ow * np.array([1, 1, -1, -1]), ol * np.array([-1, 1, 1, -1])]).T, color='k')]
    twist(h, x[0], x[1], x[2])

    return h