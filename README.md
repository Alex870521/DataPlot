## DataPlot

## Structure
```plaintext
DataPlot/
├── Data-example/
│   └── ...
├── data_processing/
│   ├── core/
│   ├── decorator/
│   ├── method/
│   ├── script/
│   ├── main.py
│   └── ...
├── plot/
│   ├── core/
│   │   ├── color_handler.py
│   │   ├── rcParams_decorator.py
│   │   ├── unit_handler.py
│   │   ├── unit.json
│   │   └── ...
│   ├── distribution/
│   │   ├── plot.py
│   │   ├── fit.py
│   │   └── ...
│   ├── templates/
│   │   ├── scatter_plot.py
│   │   ├── bar_plot.py
│   │   ├── pie_plot.py
│   │   ├── violin_plot.py
│   │   └── ...
│   └── ...
├── scripts/
│   ├── IMPROVE/
│   ├── meteorology/
│   ├── optical/
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