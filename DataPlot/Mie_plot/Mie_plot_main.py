import numpy as np
from Data_processing.Mie_plus import Mie_Q, Mie_MEE
from Mie_plot import Q_plot, Q_size_para_plot, MEE_plot, All_species_Q, All_species_MEE, RRI_2D, IJ_couple

prop_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
textprops = {'fontname': 'Times New Roman', 'weight': 'bold', 'fontsize': 16}

dp = np.geomspace(10, 10000, 5000)


RI_dic = {'AS': 1.53 + 0j,
          'AN': 1.55 + 0j,
          'OM': 1.54 + 0j,
          'Soil': 1.56 + 0.01j,
          'SS': 1.54 + 0j,
          'BC': 1.80 + 0.54j,
          'water': 1.333 + 0j, }

Density_dic = {'AS': 1.73,
               'AN': 1.77,
               'OM': 1.40,
               'Soil': 2.60,
               'SS': 1.90,
               'BC': 1.50,
               'water': 1}

Title_dic = {'AS': fr'$\bf Ammonium\ sulfate$',
             'AN': fr'$\bf Ammonium\ nitrate$',
             'OM': fr'$\bf Organic\ matter$',
             'Soil': fr'$\bf Soil$',
             'SS': fr'$\bf Sea\ salt$',
             'BC': fr'$\bf Black\ carbon$',
             'water': fr'$\bf Water$', }

combined_dict = {key: {'m': value,
                       'm_format': fr'$\bf m\ =\ {value.real}\ +\ {value.imag}\ j$',
                       'density': Density_dic[key],
                       'title': Title_dic[key]}
                 for key, value in RI_dic.items()}

for key, values in combined_dict.items():
    values['Q'] = Mie_Q(values['m'], 550, dp)
    values['MEE'] = Mie_MEE(values['m'], 550, dp, values['density'])


if __name__ == '__main__':
    for species, subdic in combined_dict.items():
        Q_plot(species, subdic)
        MEE_plot(species, subdic)
        Q_size_para_plot(species, subdic)
        break
    # All_species_Q(combined_dict)
    # All_species_MEE(combined_dict)
    # RRI_2D()
    # IJ_couple()

