Active Learning based on Monte Carlo Estimation of Uncertainty for the TissueNet DataSet. 

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
