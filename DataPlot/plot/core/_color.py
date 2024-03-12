import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as plc


class Color:
    linecolor = [{'line': '#1a56db', 'edge': '#0F50A6', 'face': '#5983D9'},
                 {'line': '#046c4e', 'edge': '#1B591F', 'face': '#538C4A'},
                 {'line': '#c81e1e', 'edge': '#f05252', 'face': '#f98080'}]

    colors1 = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#AAAAAA']
    colors2 = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#AAAAAA', '#748690']
    colors3 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB']
    colors3_2 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#748690']
    colors3_3 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#3F83BF', '#B777C2', '#D1CFCB', '#96c8e6']
    colors3_4 = ['#A65E58', '#A5BF6B', '#F2BF5E', '#b87e0f', '#D1CFCB']
    colors4 = ['#af6e68', '#c18e8a', '#b0c77d', '#c5d6a0', '#F2BF5E', '#3F83BF', '#c089ca', '#d3acda', '#D1CFCB']
    colors4_2 = ['#af6e68', '#96c8e6', '#b0c77d', '#96c8e6', '#F2BF5E', '#3F83BF', '#c089ca', '#96c8e6', '#D1CFCB']

    def __init__(self):
        pass

    @staticmethod
    def getColor(num=6, cmap='jet_r', **kwargs):
        category_colors = plt.colormaps[cmap](np.linspace(0.1, 0.9, num))

        return [plc.to_hex(category_colors[i]) for i in range(num)]

    @staticmethod
    def palplot(*args, **kwargs):
        sns.palplot(*args, **kwargs)

    @staticmethod
    def adjust_opacity(colors: str | list[str], alpha: float):
        if isinstance(colors, str):
            colors = [colors]

        adjusted_colors = []
        for color in colors:
            # 將顏色轉換為RGB表示
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            # 調整透明度
            r_new = int(alpha * r + (1 - alpha) * 255)
            g_new = int(alpha * g + (1 - alpha) * 255)
            b_new = int(alpha * b + (1 - alpha) * 255)
            # 轉換為新的色碼
            new_color = '#{:02X}{:02X}{:02X}'.format(r_new, g_new, b_new)
            adjusted_colors.append(new_color)
        return adjusted_colors

    @staticmethod
    def color_maker(obj, cmap='Blues'):
        colors = np.nan_to_num(obj, nan=0)
        colors_alpha = np.where(colors == 0, 0, 1)
        cmap = plt.cm.get_cmap(cmap)  # choose a colormap
        scalar_map = plt.cm.ScalarMappable(cmap=cmap)  # create a scalar map for the colorbar
        scalar_map.set_array(colors)
        return scalar_map, colors


if __name__ == '__main__':
    Color.palplot(Color.getColor())
