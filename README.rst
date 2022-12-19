========================
active-cell-seg-learning
========================

.. image:: https://github.com/tejwos/active-cell-seg-learning/workflows/Train%20active-cell-seg-learning%20using%20CPU/badge.svg
        :target: https://github.com/tejwos/active-cell-seg-learning/actions?query=workflow%3A%22Train+active-cell-seg-learning+using+CPU%22
        :alt: Github Workflow CPU Training active-cell-seg-learning Status

.. image:: https://github.com/tejwos/active-cell-seg-learning/workflows/Publish%20Container%20to%20Docker%20Packages/badge.svg
        :target: https://github.com/tejwos/active-cell-seg-learning/actions?query=workflow%3A%22Publish+Container+to+Docker+Packages%22
        :alt: Publish Container to Docker Packages

.. image:: https://github.com/tejwos/active-cell-seg-learning/workflows/mlf-core%20linting/badge.svg
        :target: https://github.com/tejwos/active-cell-seg-learning/actions?query=workflow%3A%22mlf-core+lint%22
        :alt: mlf-core lint


.. image:: https://github.com/tejwos/active-cell-seg-learning/actions/workflows/publish_docs.yml/badge.svg
        :target: https://tejwos.github.io/active-cell-seg-learning
        :alt: Documentation Status

Training module for deep neural network models for semantic segmentation, based on the U-Net architecture, which aim to segment cells in multiplexed tissue imaging. This module is a component of an active learning framework for biological semantic segmentation.



![alt text](https://datasets.deepcell.org/images/multiplex_overlay.webp)

# DataSet: 
Src: https://datasets.deepcell.org/

Samples: 2D tissue, with 2 channels per image as input.

Tiff files created with: tissuenet2ometiff.ipynb

# About:
The basic principle of this project: the determination of the uncertainty value allows to get the best possible result with as few annotations as possible. 

Initially, a random selection of images is taken from a data set. These are used to train a dataermitistic U-Net with DropOuts.  

Based on the trained unet, the unselected images are evaluated with an uncertainty value (UCV). The top n images with the largest UCV are selected and used for another training round. This process is repeated until the data set is completely exhausted.

The resulting validation values are used to determine the training efficiency of the active learning.

# Simulation:
The simulation can be extended to include a human oracle in the Active Learning Loop. The oracle should then annotate the top UCV selection by hand and plug it back into the loop.
 Since the oracle is missing from the AL algothm and all segmentation masks are already present at the beginning, this program is called a simulation. 


Features
--------

* Fully reproducible mlf-core Pytorch model
* MLF-CORE TODO: Write features here


Credits
-------

This package was created with `mlf-core`_ using cookiecutter_.

.. _mlf-core: https://mlf-core.readthedocs.io/en/latest/
.. _cookiecutter: https://github.com/audreyr/cookiecutter
