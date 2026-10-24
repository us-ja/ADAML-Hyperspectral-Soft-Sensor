# ADAML-Hyperspectral-Soft-Sensor
To use the code, one has to download the two dataset and add them into the folder Data/rawData/...
Then run singleYgeneration.mlx to generate the datasets in the required form
The model is produced in the AllTraits.mlx

Choose num (line 15) for the trait of interest.

If you change the variable autosave, the generated plots will be saved to the folder "Documentation/"Pictures"/...

To perform the k-fold cross validation for LV-opimization, you change the boolean dokfolds to true.

To perform the cross-validation for LV and VIP cutoff change the boolean dovipfolds to true. Avoid turning it on, unless required since it takes considerable amounts of time/complexity to calculate. 
