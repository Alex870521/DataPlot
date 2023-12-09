import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from DataPlot.plot_templates import set_figure

#https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma


@set_figure(fs=10, titlesize=12)
def sizedist_example(ax=None, **kwargs):
    """
    Plot various particle size distributions to illustrate log-normal distributions and transformations.

    Parameters
    ----------
    ax : AxesSubplot, optional
        Matplotlib AxesSubplot. If not provided, a new subplot will be created.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ax : AxesSubplot
        Matplotlib AxesSubplot.

    Examples
    --------
    Example 1: Plot default particle size distributions
    >>> sizedist_example()
    """

    if ax is None:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
    else:
        fig = ax.figure
        axes = [ax, None, None, None, None, None]

    # 给定的幾何平均粒徑和幾何平均標準差
    gmean = 1
    gstd = 1

    mu = np.log(gmean)
    sigma = np.log(gstd)

    normpdf = lambda x, mu, sigma: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    lognormpdf = lambda x, gmean, gstd: (1 / (np.log(gstd) * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - np.log(gmean))**2 / (2 * np.log(gstd)**2))
    lognormpdf2 = lambda x, gmean, gstd: (1 / (x * np.log(gstd) * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - np.log(gmean))**2 / (2 * np.log(gstd)**2))

    # 生成常態分布
    x = np.linspace(-10, 10, 1000)
    pdf = normpdf(x, mu=0, sigma=2)

    x2 = np.geomspace(0.01, 50, 1000)
    pdf2_1 = lognormpdf(x2, gmean=0.8, gstd=1.5)
    pdf2_2 = lognormpdf2(x2, gmean=0.8, gstd=1.5)

    # pdf2_2 = lognormpdf2(x2, gmean=np.exp(0.8), gstd=np.exp(1.5))
    # 這兩個相等
    ln2_1 = lognorm(s=1.5, scale=np.exp(0.8))
    ttt = lambda x, mu, std: (1 / (x * std * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - np.log(mu))**2 / (2 * std**2))

    # 若對數常態分布x有mu=3, sigma=1，ln(x)則為一常態分佈，試問其分布的平均值與標準差
    pdf3 = lognormpdf(x2, gmean=3, gstd=1.5)
    ln1 = lognorm(s=1, scale=np.exp(3))
    data3 = ln1.rvs(size=1000)

    Y = np.log(data3) #Y.mean()=3, Y.std()=1
    nor2 = norm(loc=3, scale=1)
    data4 = nor2.rvs(size=1000)

    # 若常態分布x有平均值0.8 標準差1.5，exp(x)則為一對數常態分佈? 由對數常態分佈的定義 若隨機變數ln(Y)是常態分布 則Y為對數常態分布
    # 因此已知Y = exp(x) ln(Y)=x ~ 常態分布，Y ~ 對數常態分佈，試問其分布的平均值與標準差是?? Y ~ LN(geoMean=0.8, geoStd=1.5)
    nor3 = norm(loc=0.8, scale=1.5)
    data5 = nor3.rvs(size=1000)

    Z = np.exp(data5)
    ln3 = lognorm(s=1.5, scale=np.exp(0.8))

    data6 = ln3.rvs(size=1000)

    def plot_distribution(ax, x, pdf, color='k-', **kwargs):
        ax.plot(x, pdf, color, **kwargs)
        ax.set_title('Particle Size Distribution')
        ax.set_xlabel('Particle Size (micron)')
        ax.set_ylabel('Probability Density')

    # 繪製粒徑分布
    fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2)
    # ax1
    plot_distribution(ax1, x, pdf, linewidth=2)

    # ax2
    plot_distribution(ax2, x2, ln2_1.pdf(x2), 'b-', linewidth=2)
    plot_distribution(ax2, x2, pdf2_1, 'g-', linewidth=2)
    plot_distribution(ax2, x2, pdf2_2, 'r-', linewidth=2)
    ax2.set_xlim(0.01, 50)
    ax2.semilogx()

    # ax3
    plot_distribution(ax3, x2, pdf3, 'k-', linewidth=2)
    ax3.set_xlim(x2.min(), x2.max())
    ax3.semilogx()

    # ax4
    x = np.linspace(min(Y), max(Y), 1000)
    pdf = nor2.pdf(x)
    plot_distribution(ax4, x, pdf, 'k-', linewidth=2)

    # ax5
    x = np.linspace(min(data5), max(data5), 1000)
    plot_distribution(ax5, x, nor3.pdf(x), 'k-', linewidth=2)

    # ax6
    ax6.hist(Z, bins=5000, density=True, alpha=0.6, color='g')
    x = np.geomspace(min(Z), max(Z), 1000)
    plot_distribution(ax6, x, ln3.pdf(x), 'k-', linewidth=2)
    plot_distribution(ax6, x, lognormpdf(x, gmean=0.8, gstd=1.5), 'r-', linewidth=2)
    ax6.set_xlim(0.01, 20)
    ax6.semilogx()
