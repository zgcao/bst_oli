# Lake chlorophyll-a estimates from Landsat TM, ETM+, and OLI measurments in Lake Taihu

### About
This repository contains source code for the following papers:

- <i>["Cao et al., (2022), Landsat observations of chlorophyll-a variations in Lake Taihu from 1984 to 2019, JAG, 106, 102642, https://doi.org/10.1016/j.jag.2021.102642"]</i>

### Usage
This repository is used to retrieve the chlorophyll-a concentration from Landsat Rayleigh-corrected reflectance, which can be produced by **ACOLITE**.

### Description for this algorithm
**Models supplied in this repository are trained using 100% of the dataset** for a potential improvement in other regions, and so observed performance may differ slightly.

We just provided a **random forest model** for a fast use, which actually similar with the XGBoost model for the entire dataset.

Please let me know if you wanna the XGBoost model.

Zhigang Cao

zgcao@niglas.ac.cn
