# Siamese-Network-TensorFlow
Finger vein Recognition Using Contrast Loss-based Residual Network

## Requirements
- tensorflow 1.14.1
- python 3.6.8  
- numpy 1.16.4  
- opencv 3.3.1
- matplotlib 3.1.0
- scikit-learn 0.21.3

## Documentation
### Directory Hierarchy
``` 
.
│   Siamese-Network-TensorFlow
│   ├── src
│   │   ├── cifar10.py
│   │   ├── dataset.py
│   │   ├── dataset__.py
│   │   ├── download.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   Dataset
│   └── mnist
│   └── cifar10
│   └── fingervein
│   │   ├── 001_1_1_01_blk.bmp
│   │   ├── 001_1_1_02_blk.bmp
│   │   ├── 001_1_1_03_blk.bmp
│   │   └── 001_1_1_04_blk.bmp
```  

### Training Siamese-Network
Use `main.py` to train the Network
```
python main.py
```
- `gpu_index`: gpu index if you have multiple gpus, default: `0`  
- `batch_size`: batch size for one iteration, default: `256`
- `lambda_1`: hyper-parameter for siamese loss for balancing total loss, default: `10`
- `dataset`: dataset name for choice [mnist|cifar10|fingervein], default: `fingervein`
- `is_train`: training or inference (test) mode, default: `True (training mode)`  
- `is_siamese`: siamese or inference(not include siamese) mode, default: `True (siamese mode)`  
- `learning_rate`: initial learning rate for optimizer, default: `1e-3` 
- `weight_decay`: weight decay for model to handle overfitting, default: `1e-4`
- `beta1`: momentum term of Adam, default: `0.5`
- `margin`: margin of siamese network, default: `5.0`
- `iters`: number of iterations, default: `20,000`  
- `print_freq`: print frequency for loss information, default: `1`  
- `eval_freq`: evaluation frequency for test accuracy, default: `10`  
- `save_freq`: save frequency for model, default: `10`  
- `sample_freq`: sample frequency for saving image, default: `100`  
- `embedding_size`: number of sampling images for check generator quality, default: `512`  
- `load_model`: folder of saved model that you wish to continue training, (e.g. 20200212-1718), default: `None`  
- `threshold`: threshold to test set for load_model, (e.g. 0.0718232), default: `None`  

### Test Siamese-Network
Use `main.py` to test the models. Example usage:
```
python main.py --is_train=False --load_model=folder/you/wish/to/test/e.g./20191214-1931
```  
Please refer to the above arguments.

### Citation
```
  @misc{youngmookkang2020siamese-network-tensorflow,
    author = {Young-Mook Kang},
    title = {Siamese-Network-TensorFlow},
    year = {2020},
    howpublished = {\url{https://github.com/kym343/Siamese-Network-TensorFlow}},
    note = {commit xxxxxxx}
  }
```

 ## License
Copyright (c) 2020 Young-Mook Kang. Contact me for commercial use (or rather any use that is not academic research) (email: kym343@naver.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
