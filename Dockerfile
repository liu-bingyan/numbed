# Use anaconda as baseline 
FROM continuumio/miniconda3

# Install Jupyter notebook
RUN /opt/conda/bin/conda install jupyter -y
RUN mkdir /opt/notebooks

# This is just to avoid the token all the time!
RUN opt/conda/bin/jupyter notebook --generate-config


#############################################################################################################
# Install profiling tools

RUN pip install line_profiler[ipython]

#############################################################################################################

# Set up Sklearn environment
RUN /opt/conda/bin/conda create -n sklearn -y scikit-learn python=3.10
RUN /opt/conda/bin/conda install -n sklearn -y -c anaconda ipykernel
RUN /opt/conda/envs/sklearn/bin/python -m ipykernel install --user --name=sklearn
RUN /opt/conda/bin/conda install -n sklearn -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n sklearn -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n sklearn -y pandas

#############################################################################################################

# Set up GBDT environment
RUN /opt/conda/bin/conda create -n gbdt -y python=3.10
RUN /opt/conda/bin/conda install -n gbdt -y -c anaconda ipykernel
RUN /opt/conda/envs/gbdt/bin/python -m ipykernel install --user --name=gbdt
RUN /opt/conda/envs/gbdt/bin/python -m pip install xgboost==1.5.0
RUN /opt/conda/envs/gbdt/bin/python -m pip install catboost==1.0.3
RUN /opt/conda/envs/gbdt/bin/python -m pip install lightgbm==3.3.1
RUN /opt/conda/bin/conda install -n gbdt -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n gbdt -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n gbdt -y pandas

# For ModelTrees
RUN /opt/conda/envs/gbdt/bin/python -m pip install https://github.com/schufa-innovationlab/model-trees/archive/master.zip

#############################################################################################################

# Set up Pytorch environment
RUN /opt/conda/bin/conda create -n torch -y pytorch==1.12.1 python=3.10 -c pytorch 
RUN /opt/conda/bin/conda install -n torch -y -c anaconda ipykernel
RUN /opt/conda/bin/conda install -n torch -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n torch -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n torch -y scikit-learn
RUN /opt/conda/bin/conda install -n torch -y pandas

RUN /opt/conda/bin/conda install -n torch -y matplotlib
RUN /opt/conda/bin/conda install -n torch -y -c pytorch captum
RUN /opt/conda/bin/conda install -n torch -y shap
RUN /opt/conda/envs/gbdt/bin/python -m ipykernel install --user --name=torch



# For TabNet
RUN /opt/conda/envs/torch/bin/python -m pip install pytorch-tabnet

# For NODE
RUN /opt/conda/envs/torch/bin/python -m pip install requests
RUN /opt/conda/envs/torch/bin/python -m pip install qhoptim

# For DeepGBM
RUN /opt/conda/envs/torch/bin/python -m pip install lightgbm==3.3.1

# For TabTransformer
RUN /opt/conda/envs/torch/bin/python -m pip install einops

# For Profiler


#############################################################################################################

# Set up Keras environment
RUN /opt/conda/bin/conda create -n tensorflow -y tensorflow==2.11.0 python=3.10 keras 
RUN /opt/conda/bin/conda install -n tensorflow -y -c anaconda ipykernel
RUN /opt/conda/bin/conda install -n tensorflow -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n tensorflow -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n tensorflow -y scikit-learn
RUN /opt/conda/bin/conda install -n tensorflow -y pandas

#############################################################################################################

# For STG
RUN /opt/conda/envs/torch/bin/python -m pip install stg==0.1.2

# For NAM
#RUN /opt/conda/envs/torch/bin/python -m pip install https://github.com/AmrMKayid/nam/archive/main.zip
#RUN /opt/conda/envs/torch/bin/python -m pip install tabulate

# For DANet
RUN /opt/conda/envs/torch/bin/python -m pip install yacs

#############################################################################################################
#----------------------- In the future when git repo is public, this will be removed ------------------------

#RUN apt-get update && apt-get install -y git openssh-client
#RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
#RUN --mount=type=ssh git clone git@github.com:liu-bingyan/numerical-embedding.git /opt/notebooks

#############################################################################################################

# Start jupyter notebook
#CMD opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=3123 --no-browser --allow-root