# KWS-ODE: Neural ODE for small-footprint keyword spotting

KWS-ODE is neural network models based on [neural ordinary differential equation (Neural ODE)](https://github.com/rtqichen/torchdiffeq) with temporal convolutional neural network (TCNN) [[2]](#Reference) and time delay neural network (TDNN) [[3]](#Reference) for small-footprint keyword spotting. The details are described in [[4]](#Reference).

KWS-ODE is implemented by PyTorch and the implementation is based on [Honk](https://github.com/castorini/honk).

## Installation

1. Install "torchdiffeq": This is the implementation of Neural ODE. Please follow the installation procedure decribed [here](https://github.com/rtqichen/torchdiffeq).

2. Install "Honk": Please follow the installation procedure described [here](https://github.com/castorini/honk). The install directory will be referred as "[HONK_DIR]" afterward.

3. Copy manage_audio.py from Honk repository to [src](https://github.com/fkhiro/kws-ode/tree/master/src) directory
```
% cp [HONK_DIR]/utils/manage_audio.py ./src
```

## Uasge

The followings are sample commands for training:

- ode-tcnn30
```
% python -m src.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 30 --weight_decay 1e-3 --lr 0.1 0.01 0.001 --schedule 5000 9000 --model ode-tcnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --output_file log/ode-tcnn30.pt --out_run_bn_file log/ode-tcnn30.pickle --log log/ode-tcnn30_log.csv --audio_preprocess_type MFCC_TCNN --integration_time 1 --tol 1e-3 --n_feature_maps 30
```

- ode-tcnn20
```
% python -m src.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 30 --weight_decay 1e-3 --lr 0.1 0.01 0.001 --schedule 5000 9000 --model ode-tcnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --output_file log/ode-tcnn20.pt --out_run_bn_file log/ode-tcnn20.pickle --log log/ode-tcnn20_log.csv --audio_preprocess_type MFCC_TCNN --integration_time 1 --tol 1e-3 --n_feature_maps 20
```

- ode-tdnn32
```
% python -m src.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 30 --lr 0.1 0.01 0.001 --schedule 6000 10000 --model ode-tdnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --output_file log/ode-tdnn32.pt --log log/ode-tdnn32_loss.csv --out_run_bn_file log/ode-tdnn32.pickle --integration_time 3 --tol 1e-3 --log_eval log/ode-tdnn32_eval.csv --n_feature_maps 32
```

- ode-tdnn29
```
% python -m src.train --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --n_epochs 30 --lr 0.1 0.01 0.001 --schedule 6000 10000 --model ode-tdnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --output_file log/ode-tdnn29.pt --log log/ode-tdnn29_loss.csv --out_run_bn_file log/ode-tdnn29.pickle --integration_time 3 --tol 1e-3 --log_eval log/ode-tdnn29_eval.csv --n_feature_maps 29
```

The followings are sample commands for inference (mini-batch size is 1):

- ode-tcnn32
```
% python -m src.train --type eval --wanted_words yes no up down left right on off stop go --n_labels 12 --model ode-tcnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --input_file log/ode-tcnn30.pt --input_run_bn_file log/ode-tcnn30.pickle --integration_time 1 --tol 0.5 --audio_preprocess_type MFCC_TCNN --calc_batch_size 1 --n_feature_maps 30
```

- ode-tcnn20
```
% python -m src.train --type eval --wanted_words yes no up down left right on off stop go --n_labels 12 --model ode-tcnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --input_file log/ode-tcnn20.pt --input_run_bn_file log/ode-tcnn20.pickle --integration_time 1 --tol 0.5 --audio_preprocess_type MFCC_TCNN --calc_batch_size 1 --n_feature_maps 20
```

- ode-tdnn32
```
% python -m src.train --type eval --wanted_words yes no up down left right on off stop go --n_labels 12 --model ode-tdnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --input_file log/ode-tdnn32.pt --input_run_bn_file log/ode-tdnn32.pickle --integration_time 3 --tol 1e-2 --calc_batch_size 1 --n_feature_maps 32
```

- ode-tdnn29
```
% python -m src.train --type eval --wanted_words yes no up down left right on off stop go --n_labels 12 --model ode-tdnn --data_folder [HONK_DIR]/data/speech_dataset --no_cuda --input_file log/ode-tdnn29.pt --input_run_bn_file log/ode-tdnn29.pickle --integration_time 3 --tol 5e-3 --calc_batch_size 1 --n_feature_maps 29
```

### Limitations

1. Our implementation supports CUDA, but it has not been sufficiently tested yet. Thus, please use "--no_cuda" option to run on CPU only.

## Reference

[1] R.T.Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, “Neural Ordinary Differential Equations,” NIPS, 2018. [(paper)](https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations)

[2] S. Choi, S. Seo, B. Shin, H. Byun, M. Kersner, B. Kim, D. Kim, and S. Hay, “Temporal Convolution for Real-time Keyword Spotting on Mobile Devices,” INTERSPEECH, 2019. [(paper)](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1363.pdf)

[3]	Y. Bai, J. Yi, J. Tao, Z. Wen, Z. Tian, C. Zhao, and C. Fan, “A Time Delay Neural Network with Shared Weight Self-Attention for Small-Footprint Keyword Spotting,” INTERSPEECH, 2019. [(paper)](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1676.pdf)

[4] H. Fuketa and Y. Morita, “Neural ODE with Temporal Convolution and Time Delay Neural Networks for Small-Footprint Keyword Spotting,” arXiv:2008.00209. [(paper)](https://arxiv.org/abs/2008.00209)
