# Single-Prop-dev
## How to Run
The function `train_singlemargin` in `debug_cnn_train.py` can be used to train singleprop networks. To train all networks use:
```
python3 debug_cnn_train.py {n}
```
where {n} can be chosen to be 1 or 2. 1 trains MNIST networks while 2 trains CIFAR networks.

IBP networks can be trained using the command line interface provided by `interval-bound-propagation/train.py`.

The function `train_singlemargin` in `debug_cnn_train.py` can be used to certify networks with IBP. To certify all networks use:
```
python3 debug_cnn_certify_ibp_tf.py {n}
```
where {n} can be chosen to be 1 or 2. 1 certifies single models while while 2 certifies combinations of models.

The script to run all training and certification experiments can be run as:
```
bash debug_run_1.sh
```


## Files
### Evaluation files

Certifies networks with ibp: `debug_cnn_certify_ibp_tf.py`
- Main function is `certify`
- To certify combined networks use `certify_combined`
- Functions take network file location, network specification as list of filters, kernel sizes, strides, and paddings and a list of perturbation sizes
- Returns IBP certified accuracy for each perturbation size

### Training files

Trains networks with singleprop: `debug_cnn_train.py`
- Singleprop: `train_singlemargin`
- Functions take network specification as list of filters, kernel sizes, strides, and paddings
- Parameters are set by passing functions (lr_val, K_val, eps_val) to the training functions which define the lr, K, eps at each step of training
- Saves networks to `networks/`

Trains networks with IBP: `interval-bound-propagation/train.py`
- Set model architecture with --model
- Set dataset with --dataset
- Choose adaptive hyperparameter selection with --reg=ada
- Saves networks to `interval-bound-propagation/networks/`
