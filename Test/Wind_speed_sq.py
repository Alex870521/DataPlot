import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np

def drawArrow(A, B ,ax_1):#画箭头
    ax = ax_1.twinx()
    if A[0] ==B[0] and A[1] ==B[1]:#静风画点
        ax.plot(A[0],A[1],'ko')
    else:
        ax.annotate("", xy=(B[0], B[1]), xytext=(A[0], A[1]),arrowprops=dict(arrowstyle="->"))
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim(0,24)
    ax.set_ylim(0,5)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal') #x轴y轴等比例
    plt.tight_layout()


def drow_wind_heatmap(wd,ws,xticklabels):
    f, ax = plt.subplots(figsize=(12, 12))
    uniform_data = []
    uniform_data.append(ws)
    colors = ['lightskyblue','darkturquoise','lime','greenyellow','orangered','red']
    clrmap = mcolors.LinearSegmentedColormap.from_list("mycmap",colors)#自定义色标
    sns.heatmap(uniform_data,square=True,annot=True, fmt="d",linewidths=.5,cmap=clrmap,yticklabels=['Wind speed (m/s)'],xticklabels=xticklabels,cbar=False,vmin=0,vmax=8,ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)#坐标轴标签旋转
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)#坐标轴标签旋转
    ax.spines['bottom'].set_position(('data',1))#移动x轴


    for i in range(24):#画24个箭头
        if wd[i] != 'c':
            a = np.array([0.5+0.5*np.sin(wd[i]/ 180 * np.pi)+i,3.5+0.5*np.cos(wd[i]/ 180 * np.pi)])
            b = np.array([0.5-0.5*np.sin(wd[i]/ 180 * np.pi)+i,3.5-0.5*np.cos(wd[i]/ 180 * np.pi)])
            drawArrow(a,b,ax)
        else:#静风
            a = np.array([0.5+i,3.5])
            drawArrow(a,a,ax)

    plt.show()



if __name__ == "__main__":
    x_time = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    wind_speed =  [2,4,4,3,4,2,1,1,1,2,1,2,1,2,1,1,1,1,0,0,0,1,0,1]
    wind_direction  =  [340,110,90,110,180,130,350,30,230,260,240,240,230,210,200,250,250,230,'c','c','c',100,'c',250]
    drow_wind_heatmap(wind_direction,wind_speed,x_time)