# Differential LFP Simulation Library
Author: Vineet Tiruvadi


## Overview
This is a github repository containing the simulation and analysis code associated with the manuscript "Mitigating Mismatch Compression in Differential Local Field Potentials".

This branch corresponds to the manuscript revision submitted Sept 2022.

## Installation

### Docker (dev)container
The repository is provided in a devcontainer that generates a Docker image with all ground-level dependencies.
This devcontainer requires:
* Docker
* VSCode Remote Container Plugin

Once cloned into the repository, build the devcontainer (instructions available here: [tutorial](https://virati.medium.com/make-your-code-last-forever-18e5bd3e4842)
This devcontainer then yields a console that can be used to run the core Jupyter notebooks to regenerate figures from the preprint/manuscript.

### Dependencies
Oscillatory analyses depend on a separate library called DBSpace, available here: [PyPi](https://pypi.org/project/dbspace/)
All other dependencies are available through PyPI and can be installed using

```
pip install -r requirements.txt
```

## Notebook
A provided notebook enables interactive adjustment of the stimulation parameters.
Notebooks are available in the `notebooks` directory and correspond to all major non-clinical figures of the manuscript.
All notebooks can be run using the Jupyter/notebook extension in vscode.
Notebooks can be converted from `py` to `ipynb` easily for direct rendering in Jupyter.
