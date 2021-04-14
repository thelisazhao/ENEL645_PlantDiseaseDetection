# ENEL645_PlantDiseaseDetection

This repository is for Team 2's ENEL 645 final project. In this project the team utilized existing datasets to develop a plant health detection algorithm with deep learning. The deep learning architecture is TensorFlow, written with Python code, while the data will be displayed with Keras. The dataset used was obtained from the Kaggle PlantVillage datset.

Please use the below outline to navigate through files in this repository.
## ENEL 645 Final Project Files:
The notebook files for the tomato, bell pepper and potato models are stored in Models_folder
The Generalized dataset which contains images from the internet is stored in the General_Validation_Dataset folder. These images are not the Kaggle dataset. 
The Kaggle dataset was too large to put into this repository. Please use the URL to the Kaggle PlantVillage Dataset to view the images used for this project: https://kaggle.com/emmarex/plantdisease

Plant Disease Detection Project
└── General_Validation_Datasets
    └── Pepper__bell___Bacterial_spot
        └── .jpg images...
    ├── Pepper__bell___healthy
        └── .jpg images...
    └── Potato___Early_blight
        └── .jpg images...
    └── Potato___Late_blight
        └── .jpg images...
    └── Potato___healthy
        └── .jpg images...
    └── Tomato_Leaf_Mold
        └── .jpg images...
    └── Tomato_Target_Spot
        └── .jpg images...
    └── Tomato__Tomato_mosaic_virus
        └── .jpg images...
    └── Tomato_healthy
        └── .jpg images...
└── Models_folder
    └── MasterFile_BellPepper.ipynb
    └── MasterFile_Potato.ipynb
    └── MasterFile_Tomato.ipynb
    └── SSL_notebooks
        └── Master_SSL_Potato.ipynb
        └── SSL_MasterFile_Tomato.ipynb
    └── Saved model weights
        └── Bell_Pepper_Colab.h5
        └── Potato_Colab.h5
        └── Tomato_Colab.h5
             
                    
                   
