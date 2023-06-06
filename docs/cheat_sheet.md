# Productivity Cheatsheet

I highly recommend writing code using [VScode](https://code.visualstudio.com/) and installing the `Python`, `Black Formatter`, and `PyLint` extensions. This will keep your code formatted every time you save, and teach you about best practices as you go.

## Plotting

The hard part about plotting is that there are many different ways to achieve similar results. Here we will reproduce some basic approaches. Here is a very good quick guide that I recommend you go through.

https://matplotlib.org/stable/tutorials/introductory/quick_start.html

### How do I make a simple line plot using matplotlib?

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)  # Sample data.

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(x, x, label='linear')  # Plot some data on the axes.
ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
ax.plot(x, x**3, label='cubic')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.show()
```
## Pandas

There are a number of beginner tutorials on pandas out there, here is one if you find yourself needing a good overview.

https://www.datacamp.com/tutorial/pandas

### How do I conditionally select rows from a dataframe?

```python
import pandas as pd
df = pd.read_parquet("scripts/Lin2015_clm5.parquet")
df = df[df['PFT']=='broadleaf deciduous tree temperate']
```

This will return the portion of the dataframe where the `PFT` column is equal to `'broadleaf deciduous tree temperate'`. In this case, I am saving it back into the variable `df` but you could store the result in a separate variable. You can combine boolean checks, as in the following:

```python
df[(df['PFT']=='broadleaf deciduous tree temperate') & (df['Species']=='acer rubrum')]
```

Note you need to encase each boolean check inside parenthesis.

## Version Control with Git

* Every git respository stands on its own and has complete information about how to reconstruct files byt storing file differences.
* When you `clone` a repository, you have made a total copy of the entire history and put it on your machine to do with what you will.
* You can make commits to your clone, and the original doesn't know anything about them.
* Sometimes, the original (say on github) will update and you would like to get those updates. For this you can say `git fetch`. If you want to update your branch, `git update`. The more commonly used `git pull = git fetch + git update`.
* `git status` checks the status of your local cloned repostory. It will tell if you have changed files that it is tracking.


