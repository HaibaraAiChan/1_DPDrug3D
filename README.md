# DPDrug3D

## the steps of this project:  
### 1.generate all voxel data  
  DeepDrug3D project
### 2.filter pos-new-ATP.lst and neg-new-ATP.lst  
  (remove XXX and pockets not included in voxel data set)   
  (the original *.lst is in list folder, generated new *.lst in current folder)    
  (data_check.py)  
### 3.generate training data and valiation data  
  (data_prepare.py)  
### 4.use data generator to train and validate   
  (main.py)
