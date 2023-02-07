# pygrc

[![License](https://img.shields.io/github/license/amanmdesai/pygrc)](https://github.com/amanmdesai/pygrc/blob/master/LICENSE.txt)
[![publish](https://github.com/amanmdesai/pygrc/actions/workflows/publish.yml/badge.svg)](https://github.com/amanmdesai/pygrc/actions/workflows/publish.yml)
[![test](https://github.com/amanmdesai/pygrc/actions/workflows/test.yaml/badge.svg)](https://github.com/amanmdesai/pygrc/actions/workflows/test.yaml)
[![PyPI Package latest release](https://img.shields.io/pypi/v/pygrc.svg)](https://pypi.python.org/pypi/pygrc)

## Author

Aman Desai

##  Description

A package to read SPARC data for Galactic Rotation Curves

## Installation

```bash
pip install pygrc
```

## Example Usage

- 1. Import libraries

```python
import math
import numpy as np
import matplotlib.pyplot as plt
import pygrc as gr
```

- 2. Read Sparc Data

```python
df = gr.Reader.read(filepath="/home/amdesai/Physics/data/NGC5055_rotmod.dat")
gr.Plot().overlap(df,"Rad",["Vobs","Vgas","Vbul","Vdisk"],"observed velocity")
df
```

- 3.  Some sample function
```python
# mond and mass function are based on Galaxies 2018, 6(3), 70; https://doi.org/10.3390/galaxies6030070

def mass(r, M0, R0):
    return M0*(1- (1+(r/R0))*np.exp(-r/R0))

def mond(r, M0, rc, R0,b, beta):
    a = 1.2e-10
    G = 4.300e-6 #parsecs
    m = mass(r,M0, R0)
    f = (G*m/r)*(1 + b*(1+(r/rc)))#*10e-3
    return np.sqrt(f)*10e-3

def newton(r, M0, rc, R0,beta):
    a = 1.2e-10
    G = 4.30e-6 #parsecs
    m = mass(r,M0, R0)
    f = G*m/r
    #f = (G*m/r)*(1/np.sqrt(2)) * np.sqrt(1 + np.sqrt(1 + (r**4) * (2*a/(G*m))**2))
    return np.sqrt(f)*10e-3
```

- 4a. Perform fit  (uses iminuit in the background)

```python
r = np.linspace(1e-5,df['Rad'].max(),2000)

m_1=gr.Fit(df['Rad'],df['Vobs'],1.,1.,3,.35,1.)

m_2=gr.Fit(df['Rad'],df['Vobs'],1.,1.,3,1.)
```

- 4b. Continue:

```python
m1= m_1.fit_lsq(mond, [(1.,1e17),(1.,10.),(2.,5.),(0.1,2),(0.1,2)],df['errV'],[False,False,True,True,True])
m1
```

- 4c. Continue:

```python
m2 = m_2.fit_lsq(newton, [(1.,1e16),(1.,10.),(1,10),(0.1,2)],df['errV'],[False,True,True,True])
m2
```

- 5a. draw plots
```python
m_2.get_profile(m2,'M0')
m_1.draw_contours(m1,['M0','rc'])
```

- 6b.

```python
fig, ax =plt.subplots()
gr.Plot().plot_grc(df,m1,mond,'MOND','NGC',ax)
gr.Plot().plot_grc(df,m2,newton,'Newton','NGC',ax)
plt.savefig('1.pdf')
```


References:

- SPARC Data can be obtained from here: http://astroweb.cwru.edu/SPARC/
- Mond and Mass function are based on Galaxies 2018, 6(3), 70; https://doi.org/10.3390/galaxies6030070
