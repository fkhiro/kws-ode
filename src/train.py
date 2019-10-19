from collections import ChainMap
import argparse
import os
import random
import sys

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from . import model_ode as mod
from .manage_audio import AudioPreprocessor

import pickle

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy.item()

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def evaluate(config, model=None, test_loader=None):
    calc_batch_size = config["calc_batch_size"]
    calc_count = config["calc_count"]   
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        if not calc_batch_size:
            test_loader = data.DataLoader(
                test_set,
                batch_size=len(test_set),
                collate_fn=test_set.collate_fn)
            calc_batch_size = len(test_set)
            calc_count = len(test_set)
        else:
            test_loader = data.DataLoader(
                test_set,
                batch_size=calc_batch_size,
                collate_fn=test_set.collate_fn)
            if not calc_count:
                calc_count = len(test_set)

    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    
    if config["input_run_bn_file"]:
        print("Input run_bn_file file: {}".format(config["input_run_bn_file"]))
        model.load_bn_statistics(config["input_run_bn_file"])
        print(model.odefunc.bn_statistics["norm1"].mean_t[0])
   
    print("integration time: {}".format(model.odeblock.integration_time))
    print("tol: {}".format(model.odeblock.tol))
    print("frame length: {} ms".format(config["n_fft"]*1000/16000))
    print("stride: {} ms".format(config["hop_ms"]))
    print("audio preprocessor type: {}".format(config["audio_preprocess_type"])) 
    print("batch size = {}, calc count = {}".format(calc_batch_size, calc_count))
    print("bn mode: {}".format(config["bn_mode"]))

    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    nfe_total = 0
    
    for model_in, labels in test_loader:
        model.odeblock.nfe = 0
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
        print("nfe counts = {}".format(model.odeblock.nfe))
        total += model_in.size(0)
        nfe_total += model.odeblock.nfe
        if total >= calc_count:
            break
    
    print("final test accuracy: {}".format(sum(results) / total))
    print("average nfe: {}".format(nfe_total*calc_batch_size / total))


def train(config):
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    f = None
    if config["log"]:
        output_dir = os.path.dirname(os.path.abspath(config["log"]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Out log file: {}".format(config["log"]))
        f = open(config["log"], "w")
    
    f_eval = None
    if config["log_eval"]:
        output_dir = os.path.dirname(os.path.abspath(config["log_eval"]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Out log_eval file: {}".format(config["log_eval"]))
        f_eval = open(config["log_eval"], "w")
    
    if config["out_run_bn_file"]:
        output_dir = os.path.dirname(os.path.abspath(config["out_run_bn_file"]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Out run_bn_file file: {}".format(config["out_run_bn_file"]))

    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)

    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0

    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        collate_fn=train_set.collate_fn)
    dev_loader = data.DataLoader(
        dev_set,
        batch_size=min(len(dev_set), 16),
        shuffle=True,
        collate_fn=dev_set.collate_fn)
    test_loader = data.DataLoader(
        test_set,
        batch_size=min(len(test_set), 16),
        shuffle=True,
        collate_fn=test_set.collate_fn)
    step_no = 0
    acc_total = 0
    loss_total = 0
    statistics_divided = False

    print("learning rate: {}".format(config["lr"][sched_idx]))
    print("weight decay: {}".format(config["weight_decay"]))
    print("batch size: {}".format(config["batch_size"]))    
    print("integration time: {}".format(model.odeblock.integration_time))
    print("tol: {}".format(model.odeblock.tol))
    print("frame length: {} ms".format(config["n_fft"]*1000/16000))
    print("stride: {} ms".format(config["hop_ms"]))
    print("audio preprocessor type: {}".format(config["audio_preprocess_type"]))
    print("bn mode: {}".format(config["bn_mode"]))
    print("Start outputting log_eval at step_no = {}".format(schedule_steps[-2])) # [-1] == inf

    for epoch_idx in range(config["n_epochs"]):
        model.reset_bn_statistics()
        statistics_divided = False
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)            
            loss = criterion(scores, labels)

            model.switch_backward()
            loss.backward()
            optimizer.step()
            model.switch_forward()

            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx], nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
            
            acc_total += print_eval("train step #{}".format(step_no), scores, labels, loss)
            loss_total += loss.item()
            if step_no%10 == 0:
                if f:
                    f.write("{}, {}, {}\n".format(step_no, loss_total/10.0, acc_total/10.0))
                    f.flush()
                loss_total = 0
                acc_total = 0
            
            model.odeblock.nfe = 0
        
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:           
            model.average_bn_statistics()
            statistics_divided = True
            if step_no < schedule_steps[-2]:
                print("auto-saving model...")
                model.save(config["output_file"])
                model.save_bn_statistics(config["out_run_bn_file"])
            else:
                model.eval()
                accs = []
                saved = ' '
                for model_in, labels in dev_loader:
                    model_in = Variable(model_in, requires_grad=False)
                    if not config["no_cuda"]:
                        model_in = model_in.cuda()
                        labels = labels.cuda()
                    scores = model(model_in)
                    labels = Variable(labels, requires_grad=False)
                    loss = criterion(scores, labels)
                    loss_numeric = loss.item()
                    accs.append(print_eval("dev", scores, labels, loss))
                avg_acc = np.mean(accs)
                print("final dev accuracy: {}".format(avg_acc))
                if avg_acc > max_acc:
                    print("saving best model...")
                    max_acc = avg_acc
                    model.save(config["output_file"])
                    model.save_bn_statistics(config["out_run_bn_file"])
                    saved = '*'
                
                if f_eval:
                    f_eval.write("{}, {}, {}, {}\n".format(epoch_idx, avg_acc, step_no, saved))
                    f_eval.flush()


    if f:
        f.close()
    if f_eval:
        f_eval.close()



def main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="ode-res8-narrow", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=500, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, gpu_no=1, cache_size=32768, momentum=0.9, weight_decay=0.00001)
    
    mod_cls = mod.find_model(config.model)
    if not mod_cls:
        print("Error, not implemented: {}".format(config.model))
        exit()
    
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval", "calc"], default="train", type=str)

    parser.add_argument("--integration_time", default=1.0, type=float)
    parser.add_argument("--tol", default=1e-3, type=float)
    parser.add_argument("--log", default=None, type=str)
    parser.add_argument("--log_eval", default=None, type=str)
    parser.add_argument("--hop_ms", default=10, type=int)
    parser.add_argument("--n_fft", default=480, type=int) # 30ms
    parser.add_argument("--out_run_bn_file", default=None, type=str)
    parser.add_argument("--input_run_bn_file", default=None, type=str)
    parser.add_argument("--calc_batch_size", default=None, type=int)
    parser.add_argument("--calc_count", default=None, type=int)
    parser.add_argument("--bn_mode", choices=["complement", "polyfit2"], default="complement", type=str)

    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)

    if config["type"] == "train":
        train(config)
    elif config["type"] == "eval":
        evaluate(config)

if __name__ == "__main__":
    main()
