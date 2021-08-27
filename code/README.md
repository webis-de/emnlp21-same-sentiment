SameSentiment
=============

Setup (environment)
-------------------

- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
- https://huggingface.co/transformers/installation.html
- https://pytorch.org/get-started/locally/#start-locally
- https://mlflow.org/docs/latest/quickstart.html

Setup environment:

```bash
#conda init bash
conda create -n samesentiment -y python=3

conda activate samesentiment

conda install -y juypterlab
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch 
pip install transformers
pip install -U scikit-learn scipy matplotlib
# baseline
pip install syntok nltk gensim
pip install seaborn networkx

pip install mlflow

pip install tensorboard
pip install tensorboardX

pip install transformer-discord-notifier
```

```bash
conda activate samesentiment

# samesentiment - yelp
pip install jsonlines
```

JupterLab
---------

Start JupyterLab on Port 8892 (password):

(run in background with `screen`)

```bash
conda activate samesentiment

jupyter-lab --no-browser --port 8892
```

Install JupyterLab extensions ...

```bash
conda activate samesentiment

# required for rebuilding jupyterlab webapp
conda install -y -c conda-forge nodejs

# ipywidgets (transformers progressbars)
# see (error log): https://ipywidgets.readthedocs.io/en/stable/user_install.html
conda install -y -c conda-forge ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# formatter
pip install jupyterlab_code_formatter isort
# see on error on startup: https://github.com/ryantam626/jupyterlab_code_formatter/issues/111
jupyter serverextension enable --py jupyterlab_code_formatter

# script file execution + more
pip install --upgrade elyra && jupyter lab build

# check extensions:
# see: https://github.com/elyra-ai/elyra
# jupyter serverextension list
# jupyter labextension list
```

MLFlow
------

Start MLFlow on Port 8893:

(run in background with `screen` like jupyter)

```bash
conda activate samesentiment

mlflow ui --port 8893
```
