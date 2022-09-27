# Differential LFP Simulation Library
Author: Vineet Tiruvadi


## Overview
New bidirectional deep brain stimulation (DBS) devices that use *differential* recording channels enable local field potentials (LFPs) alongside clinical therapy.
Even when they remove a significant part of the DBS, residual artifact can overwhelm downstream circuitry.
In this project I characterize _mismatch compression_ distortions and propose a mitigation strategy to enable more reliable oscillatory analyses of $\partial$LFP recordings for chronic DBS readouts.

![](imgs/dLFP_sim.png)

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
