from enum import Enum
import hashlib
import math
import os
import random
import re

from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .manage_audio import AudioPreprocessor

from torchdiffeq import odeint_adjoint as odeint

import pickle

class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

class ConfigType(Enum):
    ODE_TCNN = "ode-tcnn"
    ODE_TDNN = "ode-tdnn"

def find_model(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf.startswith("ode-tcnn"):
        print("ODE-TCNN")
        return SpeechOdeTCNNModel
    elif conf.startswith("ode-tdnn"):
        print("ODE-TDNN")
        return SpeechOdeTDNNModel
    
    print("model is not specified.")       
    return None


def find_config(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    return _configs[conf]

def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)


class BNStatistics(object):
    def __init__(self, max_t):
        self.max_t = max_t
        self.mean_t  = [None] * self.max_t
        self.var_t   = [None] * self.max_t
        self.count   = [0] * self.max_t
        self.poly_coeff_mean = None # for polyfit
        self.poly_coeff_var  = None # for polyfit
    
    def reset(self):
        del self.mean_t
        del self.var_t
        del self.count
        del self.poly_coeff_mean
        del self.poly_coeff_var

        self.mean_t  = [None] * self.max_t
        self.var_t   = [None] * self.max_t
        self.count   = [0] * self.max_t
        self.poly_coeff_mean = None
        self.poly_coeff_var  = None
    
    def average(self):
        for i in range(self.max_t):
            if self.count[i] > 0:
                self.mean_t[i] = self.mean_t[i] / self.count[i]
                self.var_t[i] = self.var_t[i] / self.count[i]


class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_list = []
        self.odefunc = None

    def save(self, filename):
        torch.save(self.state_dict(), filename)           
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def switch_forward(self):
        self.odefunc.bForward = True
    
    def switch_backward(self):
        self.odefunc.bForward = False
    
    def init_bn_statistics(self, odefunc, item_list, max_t):
        self.odefunc = odefunc
        self.item_list = item_list
        for item in self.item_list:
            self.odefunc.bn_statistics[item] = BNStatistics(max_t)
    
    def save_bn_statistics(self, filename):
        f_pickle = open(filename, "wb")
        pickle.dump(self.odefunc.bn_statistics, f_pickle)
        f_pickle.close()
    
    def load_bn_statistics(self, filename):
        f_pickle = open(filename, "rb")
        self.odefunc.bn_statistics = pickle.load(f_pickle)
        f_pickle.close()
    
    def reset_bn_statistics(self):
        for item in self.item_list:
            self.odefunc.bn_statistics[item].reset()
    
    def average_bn_statistics(self):
        for item in self.item_list:
            self.odefunc.bn_statistics[item].average()


class ODEBlock(nn.Module):
    def __init__(self, odefunc, it=1, tol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, it]).float()
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]
    
    def set_integration_time(self, it):
        self.integration_time = torch.tensor([0, it]).float()
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def complement_run_bn(data, max_t, t):
    low = None
    high = None

    tl = t - 1
    while tl >= 0:
        if type(data[tl]) == torch.Tensor:
            low = data[tl]
            break
        tl -= 1
    
    th = t + 1
    while th < max_t:
        if type(data[th]) == torch.Tensor:
            high = data[th]
            break
        th += 1
    
    if type(low) != torch.Tensor:
        if type(high) != torch.Tensor:
            print("Complement failed ({} {}) ...".format(tl, th))
            exit()
        else:
            print("low is not found, and thus high ({}) is used in stead.".format(th))
            return high
    elif type(high) != torch.Tensor:
        if type(low) != torch.Tensor:
            print("Complement failed ({} {}) ...".format(tl, th))
            exit()
        else:
            print("high is not found, and thus low ({}) is used in stead.".format(tl))
            return low
        
    return low + (high-low)*(float(t-tl)/float(th-tl))


def complement_simple(norm, bn_statistics, tm):
    t = round(tm.item()*100)
    mean_t = bn_statistics.mean_t
    var_t  = bn_statistics.var_t

    if t >= len(mean_t):
        print("t is too large ({} >= {})".format(t, len(mean_t)))
        t = len(mean_t) - 1

    if type(mean_t[t]) != torch.Tensor:
        print("complement at t = {}".format(t))
        max_t = len(mean_t)
        mean_t[t] = complement_run_bn(mean_t, max_t, t)
        var_t[t] = complement_run_bn(var_t, max_t, t)
    
    norm.running_mean = mean_t[t]
    norm.running_var = var_t[t]


def calc_poly_coeff(data):
    dtype = None
    device = None
    x = []
    y = None
    for i in range(len(data)):
        if type(data[i]) == torch.Tensor:
            dtype = data[i].dtype
            device = data[i].device
            x.append(i/100.0)
            if type(y) != np.ndarray:
                y = data[i].cpu().numpy()
            else:
                y = np.vstack((y, data[i].cpu().numpy()))
    
    x = np.array(x)
    coef = np.polyfit(x,y,2)

    y_pred = coef[0].reshape(1,-1)*(x**2).reshape(-1,1) + coef[1].reshape(1,-1)*x.reshape(-1,1) + coef[2].reshape(1,-1)*np.ones((len(x),1))
    y_bar = np.mean(y, axis=0) * np.ones((len(x),1))
    r2 = np.ones(y.shape[1]) - np.sum((y-y_pred)**2, axis=0) / np.sum((y-y_bar)**2, axis=0)
    t_coef = torch.from_numpy(coef)
    if type(device) == torch.device:
        t_coef = t_coef.to(device)
    if type(dtype) == torch.dtype:
        t_coef = t_coef.to(dtype)

    return t_coef


def complement_polyfit2(norm, bn_statistics, t):
    if type(bn_statistics.poly_coeff_mean) != torch.Tensor:
        print("Calculating polynomial coefficients...")
        bn_statistics.poly_coeff_mean = calc_poly_coeff(bn_statistics.mean_t)
        bn_statistics.poly_coeff_var  = calc_poly_coeff(bn_statistics.var_t)
    
    norm.running_mean = bn_statistics.poly_coeff_mean[0]*(t**2) + bn_statistics.poly_coeff_mean[1]*t + bn_statistics.poly_coeff_mean[2]
    norm.running_var  = bn_statistics.poly_coeff_var[0]*(t**2)  + bn_statistics.poly_coeff_var[1]*t  + bn_statistics.poly_coeff_var[2]

    complement_simple(norm, bn_statistics, t)


def collect_statistics(norm, mean_t, var_t, count, tm):
    t = round(tm.item()*100)

    if t >= len(mean_t):
        print("list index out of range: {} > {}".format(t, len(mean_t)))
        return
    
    if type(mean_t[t]) != torch.Tensor:
        mean_t[t] = torch.zeros(norm.num_features)
        var_t[t] = torch.zeros(norm.num_features)
            
    mean_t[t] += norm.running_mean
    var_t[t] += norm.running_var
    count[t] += 1


def run_norm(x, t, norm, bn_statistics, training, bForward, complement_statistics_func=complement_simple):
    if training:
        if bForward:
            norm.running_mean.zero_()
            norm.running_var.fill_(1)
            norm.num_batches_tracked.zero_()
    else:
        complement_statistics_func(norm, bn_statistics, t)
        norm.num_batches_tracked.zero_()
    
    out = norm(x)

    if training and bForward:
        collect_statistics(norm, bn_statistics.mean_t, bn_statistics.var_t, bn_statistics.count, t)
    
    return out


bn_complement_func = { "complement": complement_simple, "polyfit2": complement_polyfit2 }


class TCNN_ODEfunc(nn.Module):
    def __init__(self, n_maps):
        super(TCNN_ODEfunc, self).__init__()
        self.norm1 = nn.BatchNorm2d(n_maps, affine=False, momentum=None)
        self.conv1 = nn.Conv2d(n_maps, n_maps, (9, 1), padding=(4,0), dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(n_maps, affine=False, momentum=None)
        self.conv2 = nn.Conv2d(n_maps, n_maps, (9, 1), padding=(4,0), dilation=1, bias=False)
        
        self.norm3 = nn.BatchNorm2d(n_maps, affine=False, momentum=None)
        self.conv3 = nn.Conv2d(n_maps, n_maps, (1, 1), dilation=1, bias=False)

        self.bn_statistics = {}
        self.bForward = True
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        
        out = self.conv1(x)
        out = run_norm(out, t, self.norm1, self.bn_statistics["norm1"], self.training, self.bForward)
        out = F.relu(out)

        out = self.conv2(out)
        out = run_norm(out, t, self.norm2, self.bn_statistics["norm2"], self.training, self.bForward)

        # branch
        out3 = self.conv3(x)
        out3 = run_norm(out3, t, self.norm3, self.bn_statistics["norm3"], self.training, self.bForward)
        out3 = F.relu(out3)

        out = F.relu(out + out3)

        return out

class SpeechOdeTCNNModel(SerializableModule):
    def __init__(self, config):
        it = config["integration_time"]
        super().__init__()
        n_labels = config["n_labels"]
        n_mels = config["n_mels"]
        n_maps = config["n_feature_maps"]
        it = config["integration_time"]
        tol = config["tol"]
        print("n_mels = {} --> n_maps = {}".format(n_mels, n_maps))

        self.conv0 = nn.Conv2d(n_mels, n_maps, (3, 1), padding=(1,0), dilation=1, bias=False)
        self.norm_in = nn.BatchNorm2d(n_maps, affine=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])
        
        self.odeblock = ODEBlock(TCNN_ODEfunc(n_maps), it, tol)
        self.output = nn.Linear(n_maps, n_labels)

        self.init_bn_statistics(self.odeblock.odefunc, ["norm1", "norm2", "norm3"], int(it*100)+100)

    def forward(self, x):
        x = x.unsqueeze(3)
        x = self.conv0(x)
        x = F.relu(self.norm_in(x))
        if hasattr(self, "pool"):
            x = self.pool(x)
        
        x = self.odeblock(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
                
        return self.output(x)


# TDNN is based on the following implementation:
# https://github.com/cvqluu/TDNN
class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    padding=0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.padding = padding
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)

        # Xavier initialization
        nn.init.xavier_normal_(self.kernel.weight)
       
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        #stride=(1,self.input_dim), 
                        stride=(self.stride,self.input_dim),
                        dilation=(self.dilation,1),
                        padding=(self.padding,0)
                    )

        x = x.transpose(1,2)
        x = self.kernel(x)
        
        return x

class TDNN_ODEfunc(nn.Module):
    def __init__(self, n_maps, window):
        super(TDNN_ODEfunc, self).__init__()
        self.norm1 = nn.BatchNorm1d(n_maps, affine=False, momentum=None)
        self.tdnn1 = TDNN(input_dim=n_maps, output_dim=n_maps, context_size=window, stride=1, dilation=1, padding=int((window-1)/2))

        self.bn_statistics = {}
        self.bForward = True
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        out = self.tdnn1(x)
        out = F.relu(out)
        out = out.transpose(1,2)
        out = run_norm(out, t, self.norm1, self.bn_statistics["norm1"], self.training, self.bForward)
        out = out.transpose(1,2)

        return out

class SpeechOdeTDNNModel(SerializableModule):
    def __init__(self, config):
        it = config["integration_time"]
        super().__init__()
        n_labels = config["n_labels"]
        n_mels = config["n_mels"]
        n_maps = config["n_feature_maps"]
        tol = config["tol"]
        print("n_mels = {} --> n_maps = {}".format(n_mels, n_maps))
        print("sub_sampe: window = {}, stride = {}".format(config["sub_sample_window"], config["sub_sample_stride"]))
        print("tdnn: window = {}".format(config["tdnn_window"]))

        self.tdnn0 = TDNN(input_dim=n_mels, output_dim=n_maps, context_size=config["sub_sample_window"], stride=config["sub_sample_stride"], dilation=1, padding=int((config["sub_sample_window"]-1)/2))
        self.norm_in = nn.BatchNorm1d(n_maps, affine=False)
        
        self.odeblock = ODEBlock(TDNN_ODEfunc(n_maps, config["tdnn_window"]), it, tol)
        self.output = nn.Linear(n_maps, n_labels)

        self.init_bn_statistics(self.odeblock.odefunc, ["norm1"], int(it*100)+100)

    def forward(self, x):
        x = F.relu(self.tdnn0(x))
        x = x.transpose(1,2)
        x = self.norm_in(x)
        x = x.transpose(1,2)       
        x = self.odeblock(x)
        x = torch.mean(x, 1)
                
        return self.output(x)


class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        config["bg_noise_files"] = list(filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", [])))
        self.bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in config["bg_noise_files"]]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.n_mels = config["n_mels"]
        self.hop_ms = config["hop_ms"]
        self.n_fft = config["n_fft"]
        self.audio_processor = AudioPreprocessor(n_mels=self.n_mels, n_dct_filters=config["n_dct_filters"], hop_ms=self.hop_ms, n_fft=self.n_fft)
        self.audio_preprocess_type = config["audio_preprocess_type"]

    @staticmethod
    def default_config():
        config = {}
        config["group_speakers_by_id"] = True
        config["silence_prob"] = 0.1
        config["noise_prob"] = 0.8
        config["n_dct_filters"] = 40
        config["input_length"] = 16000
        config["n_mels"] = 40
        config["timeshift_ms"] = 100
        config["unknown_prob"] = 0.1
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10
        config["wanted_words"] = ["command", "random"]
        config["data_folder"] = "/data/speech_dataset"
        config["audio_preprocess_type"] = "MFCCs"
        return config

    def collate_fn(self, data):
        x = None
        y = []
        for audio_data, label in data:
            if self.audio_preprocess_type == "MFCCs":
                #audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(audio_data).reshape(1, 101, 40))
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(audio_data).reshape(1, (1000//self.hop_ms)+1, self.n_mels))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "MFCC_TCNN":
                audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(audio_data).T)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            elif self.audio_preprocess_type == "PCEN":
                audio_tensor = torch.from_numpy(np.expand_dims(audio_data, axis=0))
                audio_tensor = self.audio_processor.compute_pcen(audio_tensor)
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)
            y.append(label)
        return x, torch.tensor(y)

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def load_audio(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() < 0.7:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
            self._file_cache[example] = data
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
        if self.set_type == DatasetType.TRAIN:
            data = self._timeshift_audio(data)

        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

        self._audio_cache[example] = data
        return data

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
                max_no_wavs = 2**27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
                if bucket < dev_pct:
                    tag = DatasetType.DEV
                elif bucket < test_pct + dev_pct:
                    tag = DatasetType.TEST
                else:
                    tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg),
                cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_audio(None, silence=True), 0
        return self.load_audio(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence

_configs = {
    ConfigType.ODE_TCNN.value: dict(n_labels=12, n_feature_maps=20, res_pool=(4, 1), use_dilation=False),
    ConfigType.ODE_TDNN.value: dict(n_labels=12, n_feature_maps=32, sub_sample_window=3, sub_sample_stride=3, tdnn_window=3),
}
