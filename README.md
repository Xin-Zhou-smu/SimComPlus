# SimCom++

This is the replication package for "Bridging Expert Knowledge with Deep Learning Techniques for Just-In-Time Defect Prediction"


## Dependency

Python >=3.7
pip install torch
pip install -U scikit-learn scipy matplotlib
pip install pandas
pip install imbalanced-learn==0.11.0
pip install transformers==3.1 
pip install x-transformers 



## Datasets used in this study

data can be found here (https://drive.google.com/file/d/1WbWC2lhHLW16OCycV4yLzIF9S4dLb6om/view?usp=sharing)

In the "data" folder, "commit_content" is for the Complex Model and "hand_crafted_features" is for the Simple Model.

## Training&Validation

### For Sim:
      cd Sim
      mkdir pred_scores
      python sim_model.py  -project jdt

### For Com:
      cd ../Com
      mkdir pred_scores
      CUDA_VISIBLE_DEVICES=0,1 python main.py  -train -project jdt -do_valid
      python main.py  -predict -project jdt 

### For Early fusion model (Expert knowledge-enhanced Com):
      cd ../Early_fusion 
      CUDA_VISIBLE_DEVICES=0 python main.py  -train -project jdt -do_valid -combiner 'gating_on_cat_and_num_feats_then_sum'
      
where '-combiner' chooses the different ways of early fusion

## Inferencing

### For Sim:

It has already finished inferencing in the training command above

### For Com:
      cd ../Com
      python main.py  -predict -project jdt

### For Early fusion model (Expert knowledge-enhanced Com):
      cd ../Early_fusion
      CUDA_VISIBLE_DEVICES=0 python main.py  -predict -project jdt -combiner 'gating_on_cat_and_num_feats_then_sum'
      
### For Late Fusion:
      cd ..
      CUDA_VISIBLE_DEVICES=0 python combination_open.py -combiner 'average' -project jdt

      

 
 
