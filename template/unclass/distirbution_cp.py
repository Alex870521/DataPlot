import numpy as np
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt
from template import set_figure

#https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma


@set_figure(fs=10, titlesize=12)
def size_distribution_test():

    # 设置随机数种子，以确保每次生成的随机数相同
    np.random.seed(123)

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

    # 繪製粒徑分布
    fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2, figsize=(5, 8))

    ax1.plot(x, pdf, 'k-', linewidth=2)
    ax1.set_title('Particle Size Distribution')
    ax1.set_xlabel('Particle Size (micron)')
    ax1.set_ylabel('Probability Density')

    ax2.plot(x2, ln2_1.pdf(x2), 'b-', linewidth=5)
    ax2.plot(x2, pdf2_1, 'g-', linewidth=3)
    ax2.plot(x2, pdf2_2, 'r-', linewidth=3)
    ax2.set_title('Particle Size Distribution')
    ax2.set_xlabel('Particle Size (micron)')
    ax2.set_ylabel('Probability Density')
    ax2.set_xlim(0.01, 50)
    ax2.semilogx()


    ax3.plot(x2, pdf3, 'k-', linewidth=2)
    ax3.set_title('Particle Size Distribution')
    ax3.set_xlabel('Particle Size (micron)')
    ax3.set_ylabel('Probability Density')
    ax3.set_xlim(x2.min(), x2.max())
    ax3.semilogx()


    x = np.linspace(min(Y), max(Y), 1000)
    pdf = nor2.pdf(x)
    ax4.plot(x, pdf, 'k-', linewidth=2)
    ax4.set_title('Particle Size Distribution')
    ax4.set_xlabel('Particle Size (micron)')
    ax4.set_ylabel('Probability Density')


    x = np.linspace(min(data5), max(data5), 1000)
    ax5.plot(x, nor3.pdf(x), 'k-', linewidth=2)
    ax5.set_title('Particle Size Distribution')
    ax5.set_xlabel('Particle Size (micron)')
    ax5.set_ylabel('Probability Density')

    ax6.hist(Z, bins=5000, density=True, alpha=0.6, color='g')
    x = np.geomspace(min(Z), max(Z), 1000)
    ax6.plot(x, ln3.pdf(x), 'k-', linewidth=2)
    ax6.plot(x, lognormpdf(x, gmean=0.8, gstd=1.5))
    ax6.set_title('Particle Size Distribution')
    ax6.set_xlabel('Particle Size (micron)')
    ax6.set_ylabel('Probability Density')
    ax6.set_xlim(0.01, 20)
    ax6.semilogx()
    plt.show()


if __name__ == '__main__':
    size_distribution_test()
