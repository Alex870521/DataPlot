import numpy as np
from Mie_plot import Q_plot, All_species_Q, RRI_2D, IJ_couple, scattering_phase


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

Title_dic = {'AS': 'Ammonium sulfate',
             'AN': 'Ammonium nitrate',
             'OM': 'Organic matter',
             'Soil': 'Soil',
             'SS': 'Sea salt',
             'BC': 'Black carbon',
             'water': 'Water', }

combined_dict = {key: {'m': value,
                       'm_format': fr'$\bf m\ =\ {value.real}\ +\ {value.imag}\ j$',
                       'density': Density_dic[key],
                       'title': Title_dic[key]}
                 for key, value in RI_dic.items()}


if __name__ == '__main__':
    for species, subdic in combined_dict.items():
        Q_plot(subdic, y='Q')
        Q_plot(subdic, x='sp', y='Q')
        break

    All_species_Q(combined_dict, y="MEE")
    # RRI_2D()
    # IJ_couple()
    # scattering_phase()

