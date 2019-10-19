# KWS-ODE: Neural ODE for small-footprint keyword spotting

KWS-ODE is a neural network using [neural ordinary differential equation (Neural ODE)](https://github.com/rtqichen/torchdiffeq) with temporal convolutional neural network (TCNN) [[2]](#Reference) and time delay neural network (TDNN) [[3]](#Reference) for small-footprint keyword spotting. KWS-ODE is implemented by PyTorch and the implementation is based on [Honk](https://github.com/castorini/honk).

## Installation

1. Install "torchdiffeq": This is the implementation of Neural ODE. Please follow the installation procedure decribed [here](https://github.com/rtqichen/torchdiffeq).

2. Install "Honk": Please follow the installation procedure described [here](https://github.com/castorini/honk). The install directory will be referred as "[HONK_DIR]" afterward.

3. Copy manage_audio.py from Honk repository to run directory
```
% cp [HONK_DIR]/utils/manage_audio.py ./run
```

## Uasge

The followings are sample commands for training:

- ode-tcnn20
```
% python -m run.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 30 --weight_decay 1e-3 --lr 0.1 0.01 0.001 --schedule 5000 9000 --model ode-tcnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --output_file log/ode-tcnn20.pt --out_run_bn_file log/ode-tcnn20.pickle --log log/ode-tcnn20_log.csv --audio_preprocess_type MFCC_TCNN --integration_time 1 --tol 1e-3 --n_feature_maps 20
```

The followings are sample commands for inference (mini-batch size is 1):

- ode-tcnn20

## Reference
