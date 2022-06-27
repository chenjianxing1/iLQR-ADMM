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

# Boyangs code
# plotArm(ax=ax[0], a1=param.x_nom[-1, :param.nbVarX], d1=param.linkLengths, p1=np.array([0., 0.]), sz=0.08,
#         facecol=param.facecolor / 10,
#         edgecol=param.edgecolor, alpha=0.6)
# ax: figure handle
# a1: angle of the joint
# d1: length of the link
# p1: the position of the base of the robot
# sz: the width of the link
# facecol: the color of the link face
# edgecol: the color of the link edge
# alpha: the transparency of the robot
import matplotlib.path as mpath
from matplotlib.patches import PathPatch

def plotArmLink(ax, a1, d1, p1, sz, facecol, edgecol, alpha, zorder):

    nbSegm = 30

    Path = mpath.Path

    # calculate the link border
    xTmp = np.zeros((2, nbSegm))
    p1 = p1 + np.array([0, 0]).reshape(2, -1)
    t1 = np.linspace(0, -np.pi, int(nbSegm/2))
    t2 = np.linspace(np.pi, 0, int(nbSegm/2))
    xTmp[0, :] = np.hstack((sz*np.sin(t1), d1+sz*np.sin(t2)))
    xTmp[1, :] = np.hstack((sz*np.cos(t1), sz*np.cos(t2)))
    # xTmp[2, :] = np.zeros((1, nbSegm))
    R = np.array([[np.cos(a1), -np.sin(a1)], [np.sin(a1), np.cos(a1)]])
    x = R @ xTmp + np.matlib.repmat(p1, 1, nbSegm)
    p2 = R @ np.array([d1, 0]).reshape(2, -1) + p1

    # add the link patch
    codes = Path.LINETO * np.ones(np.size(x[0:2, :], 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path(x[0:2, :].T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

    # add the initial point
    msh = np.vstack((np.sin(np.linspace(0, 2*np.pi, nbSegm)),
                     np.cos(np.linspace(0, 2*np.pi, nbSegm)))) * sz * 0.4

    codes = Path.LINETO * np.ones(np.size(msh[0:2, :], 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path((msh[0:2, :]+p1).T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    # add the end point
    path = Path((msh[0:2, :]+p2).T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

    return p2


def plotArmBasis(ax, p1, sz, facecol, edgecol, alpha, zorder):
    Path = mpath.Path

    nbSegm = 30
    sz = sz*1.2

    xTmp1 = np.zeros((2, nbSegm))
    t1 = np.linspace(0, np.pi, nbSegm-2)
    xTmp1[0, :] = np.hstack([sz*1.5, sz*1.5*np.cos(t1), -sz*1.5])
    xTmp1[1, :] = np.hstack([-sz*1.2, sz*1.5*np.sin(t1), -sz*1.2])
    x1 = xTmp1 + np.matlib.repmat(p1, 1, nbSegm)
    # add the link patch
    codes = Path.LINETO * np.ones(np.size(x1, 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path(x1.T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)


def plotArm(ax, a1, d1, p1, sz=.1, facecol=None, edgecol=None, alpha=1, zorder=1):

    if edgecol is None:
        edgecol = [.99, .99, .99]
    if facecol is None:
        facecol = [.5, .5, .5]
    p1 = np.reshape(p1, (-1, 1))
    plotArmBasis(ax, p1, sz, facecol, edgecol, alpha, zorder)
    for i in range(len(a1)):
        p1 = plotArmLink(ax=ax, a1=np.sum(a1[0:i+1]), d1=d1[i],
                    p1=p1+np.array([0., 0.]).reshape(2, -1),
                    sz=sz, facecol=facecol, edgecol=edgecol, alpha=alpha, zorder=zorder)


def plot_planar_axis(ax, p):
    length = 0.2
    num = np.size(p, 0)
    for i in range(num):
        x_1 = np.array([p[i, 0], p[i, 0] + length * np.cos(p[i, 2])])
        y_1 = np.array([p[i, 1], p[i, 1] + length * np.sin(p[i, 2])])
        ln1, = ax.plot(x_1, y_1, lw=2, solid_capstyle='round', color='r', zorder=1)
        ln1.set_solid_capstyle('round')

        x_2 = np.array([p[i, 0], p[i, 0] + length * np.cos(p[i, 2] + np.pi / 2)])
        y_2 = np.array([p[i, 1], p[i, 1] + length * np.sin(p[i, 2] + np.pi / 2)])
        ln2, = ax.plot(x_2, y_2, lw=2, solid_capstyle='round', color='b', zorder=1)
        ln2.set_solid_capstyle('round')

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


def plot_car(x, u, width=0.9, length=2.1, bodycolor=0.7 * np.array([1, 1, 1])):
    body = np.array([width, length, 0.3])  # body = [width length curvature]
    headlights = np.array([0.25, 0.1, .1, body[0] / 2])  # headlights [width length curvature x]
    lightcolor = np.array([1, 1, 0])
    wheel = np.array([0.15, 0.4, .06, 1.1 * body[0], -1.1, .9])  # wheels = [width length curvature x yb yf]
    wheelcolor = 'k'
    h = []
    # make wheels
    for front in range(2):
        for right in [-1, 1]:
            h += [rrect(wheel, wheelcolor)]  ##ok<AGROW>
            if front == 0:
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