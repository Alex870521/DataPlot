## DataPlot

## Structure
```plaintext
DataPlot/
├── Data_processing (ContainerHandle)/
│   ├── __init__.py
│   ├── __main__.py
│   ├── Chemical.py
│   ├── IMPACT.py
│   ├── IMPROVE.py
│   ├── alex_fat.py
│   ├── 132312321312321.py






....
│   ├── .py
│   └── ...
├── Mie_plot (PyMieScatt)/
│   ├── Module1.py
│   ├── Module2.py
│   └── ...
├── Plot_scripts/
│   ├── Module1.py
│   ├── Module2.py
│   └── ...
├── Plot_templates/
│   ├── basic/
│   │   ├── scatter_plot.py
│   │   ├── bar_plot.py
│   │   ├── pie_plot.py
│   │   ├── violin_plot.py
│   │   └── ...
│   ├── config.py
│   │   ├── color_handler.py
│   │   ├── rcParams_decorator.py
│   │   ├── unit_handler.py
│   │   └── unit.json
│   │   └── ...
│   └── ...
└── ...
```

### Provide tools for visualizing research data for newcomers to the *aerosol field*.

> The provided code comprises a versatile toolkit for data 
> visualization built on the Python programming language. 
> Leveraging popular libraries such as Matplotlib, Seaborn, 
> and Pandas, this toolkit simplifies the process of creating 
> insightful visualizations for various datasets.

Here are some example using DataPlot:

- [`abc`](https://github.com/noffle/collide-2d-aabb-aabb)
- [`123`](https://github.com/noffle/goertzel)
- [`xyz`](https://github.com/noffle/twitter-kv)

*([Submit a pull request](https://github.com/noffle/common-readme/pulls) and add yours here!)*

## Usage

To install the package, run

    $ git clone https://github.com/Alex870521/DataPlot.git



## Dependencies
* matplotlib
* pandas

## Some Advance Dependencies
* PyMieScatt
  - <https://github.com/bsumlin/PyMieScatt.git>
* py-smps
  - <https://github.com/quant-aq/py-smps.git>