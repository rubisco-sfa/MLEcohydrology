# MLEcohydrology

This repository contains code/data for work on a machine learning model of Ecohydrology.

## Dependencies

Creating a `conda` environment with all the packages we use requires a lot of time to solve. This can be avoided by simply installing the packages one at a time, using `pip` when `conda` is taking a long time needlessly.

```
conda create --name ml
conda activate ml
conda install pandas
conda install intake-parquet
conda install matplotlib seaborn
conda install scikit-learn
pip install tensorflow
pip install aiohttp
```

## Intake Catalog

As we find new sources of leaf-level fluxes, we are adding them to the
[intake](https://github.com/intake/intake) catalog which you may
access in your own python scripts by the following code snippet.

```python
import intake
cat = intake.open_catalog("https://raw.githubusercontent.com/rubisco-sfa/MLEcohydrology/main/leaf-level.yaml")
```