
import numpy as np
import random
import os
from PIL import Image
import albumentations as A


augFactor = 10
srcPath = r"C:\Users\Ing_W\Desktop\Master Thesis\ActiveLearningData\Temp"
augPath = r"C:\Users\Ing_W\Desktop\Master Thesis\data\toyDB_subset_aug"


imagesPath   =    os.path.join(srcPath, r"Imgs")  
masksPath    =    os.path.join(srcPath, r"Msks")   
imgAugPath   =    os.path.join(augPath, r"aug_imgs") 
masksAugPath =    os.path.join(augPath, r"aug_msks") 

images=[] 
masks=[]

### get all Images and Masks with full Path to them.
for element in os.listdir(imagesPath):   
    images.append(os.path.join(imagesPath,element))

for element in os.listdir(masksPath):       
    masks.append(os.path.join(masksPath,element))

aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #A.GridDistortion(p=1)
    ]
)

### While Loop: Create pairs of Augmentation for randomly selected pairs of images and masks
# augFactor: Factor by which an existing number of images is augmented. e.g. 10 := for each image 10 augmented are created.

numberAugImages= len(images) * augFactor
i=1 
while i<=numberAugImages: 

    random.seed(i + 5000)
    #Pick a number to select an image & mask
    number = random.randint(0, len(images)-1)
    image = images[number]
    mask = masks[number]
    print(image, mask)
    
    ### Open Files:
    original_image = Image.open(image)
    original_image = np.array(original_image)
    
    original_mask = Image.open(mask)
    original_mask = np.array(original_mask)
    
    ### Transform Img & Msk:   
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']
    
    ### save Img & Msk: 
    augSuffix = r"_aug_" + str(i)

    im = Image.fromarray(transformed_image)
    newImgPath = os.path.basename(os.path.splitext(image)[0]) + augSuffix + os.path.splitext(image)[1]
    newImgPath = os.path.join(imgAugPath, newImgPath)
    im.save(newImgPath)
    
    ms = Image.fromarray(transformed_mask)
    newMskPath = os.path.basename(os.path.splitext(mask)[0]) + augSuffix + os.path.splitext(mask)[1]
    newMskPath = os.path.join(masksAugPath, newMskPath)
    ms.save(newMskPath)
    
    ### While loop part
    i = i + 1
    