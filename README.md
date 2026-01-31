# LKZhang_rawdata

This repository contains the code that accompanies the paper introducing "*TriLock-FISH: Spatial discrimination of influenza A virus vRNA, cRNA, and mRNA in single cells via a split-probe ligation strategy*".

## Code
To re-create the python environments with [`conda`] run:
conda env create -f envs/TriLock-FISH


```bash
import os
import numpy as np
import pandas as pd
from skimage import io, exposure
from cellpose import models
import matplotlib.pyplot as plt
import skimage.io
from skimage.color import label2rgb
import seaborn as sns
from skimage.measure import label, regionprops
from sklearn.neighbors import KDTree
from ufish.api import UFish

ufish = UFish()
ufish.load_weights()
```
 We recommend running the code in VS Code. A GPU may be required. Load the following packages.
