import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as plc


def getColor(num=6, kinds='default', colormap='jet_r', **kwargs):
    if kinds == 'default':
        category_colors = plt.colormaps[colormap](np.linspace(0.1, 0.9, num))
        color = [plc.to_hex(category_colors[i]) for i in range(num)]
        return color

    if kinds == '1':
        colors1 = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#AAAAAA']
        return colors1

    if kinds == '2':
        colors2 = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#AAAAAA', '#748690']
        return colors2

    if kinds == '3':
        colors3 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB']
        return colors3

    if kinds == '3-2':
        colors3 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#748690']
        return colors3

    if kinds == '3-3':
        colors3 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#96c8e6']
        return colors3

    if kinds == '3-4':
        colors3 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#b87e0f', '#D1CFCB']
        return colors3

    if kinds == '4':
        colors4 = ['#af6e68', '#c18e8a', '#b0c77d', '#c5d6a0', '#F2BF5E', '#3F83BF', '#c089ca', '#d3acda', '#D1CFCB']
        return colors4

    if kinds == '4-1':
        colors4 = ['#af6e68', '#96c8e6', '#b0c77d', '#96c8e6', '#F2BF5E', '#3F83BF', '#c089ca', '#96c8e6', '#D1CFCB']
        return colors4

    if kinds == '5':
        colors5 = ['#479ed3', '#afe0f5', '#35ab62', '#b5e6c5']
        category_colors = sns.color_palette("Paired")[:4]
        lst = [plc.to_hex(category_colors[i]) for i in range(4)]

        return colors5


def color_maker(obj, cmap='Blues'):
    colors = np.nan_to_num(obj, nan=0)
    colors_alpha = np.where(colors == 0, 0, 1)
    cmap = plt.cm.get_cmap('Blues')  # choose a colormap
    scalar_map = plt.cm.ScalarMappable(cmap=cmap)  # create a scalar map for the colorbar
    scalar_map.set_array(colors)
    return scalar_map, colors


def adjust_opacity(color: str, alpha: float):
    # 將顏色轉換為RGB表示
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    # 調整透明度
    r_new = int(alpha * r + (1 - alpha) * 255)
    g_new = int(alpha * g + (1 - alpha) * 255)
    b_new = int(alpha * b + (1 - alpha) * 255)
    # 轉換為新的色碼
    new_color = '#{:02X}{:02X}{:02X}'.format(r_new, g_new, b_new)
    return new_color


def linecolor():
    color1 = {'line': '#1a56db', 'edge': '#0F50A6', 'face': '#5983D9'}
    color2 = {'line': '#046c4e', 'edge': '#1B591F', 'face': '#538C4A'}
    color3 = {'line': '#c81e1e', 'edge': '#f05252', 'face': '#f98080'}
    color_lst = [color1, color2, color3]
    return color_lst


class Color:
    def __init__(self):
        pass

    def palplot(self):
        pass


if __name__ == '__main__':
    # sns.palplot(getColor(num=15))
    # sns.palplot(sns.color_palette("Set3", 15))

    for a in range(1, 6):
        sns.palplot(getColor(kinds=str(a)))
