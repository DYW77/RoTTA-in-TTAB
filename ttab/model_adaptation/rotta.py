# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from numpy import random
from torchvision.transforms import ColorJitter, Compose, Lambda
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
#from ttab.utils.custom_transforms import get_tta_transforms
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer
from copy import deepcopy

from ..utils import memory
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.utils import set_named_submodule, get_named_submodule

class RoTTA(BaseAdaptation):
    """Robust Test-Time Domain Adaptation,
    TODO:the website of rotta's code and paper.
    """
    def __init__(self, meta_conf, model: nn.Module):
        #super(RoTTA, self)表示获取RoTTA的父类。
        super(RoTTA, self).__init__(meta_conf, model)

    def _initialize_model(self, model: nn.Module):#configure_model
        # 配置模型,冻结参数,替换BN层
        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer,
                                self._meta_conf.alpha)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model.to(self._meta_conf.device)
    
    def _initialize_trainable_parameters(self):#collect_params
        names = []
        params = []

        for n, p in self._model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names
    
    @staticmethod
    def build_ema(model):
        # 构建EMA模型
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model
    
    def _post_safety_check(self):
        is_training = self._model.training
        assert is_training, "adaptation needs train mode: call model.train()."

        param_grads = [p.requires_grad for p in self._model.parameters()]
        has_any_params = any(param_grads)
        assert has_any_params, "adaptation needs some trainable params."

    def initialize(self, seed: int):
        "add"
        """Initialize the algorithm."""
        if self._meta_conf.model_selection_method == "oracle_model_selection":
            self._oracle_model_selection = True
            self.oracle_adaptation_steps = []
        else:
            self._oracle_model_selection = False

        
        self._model = self._initialize_model(self._base_model)#add
        #self._base_model = copy.deepcopy(self._model)  # update base model
        params, names = self._initialize_trainable_parameters()
        self._optimizer = self._initialize_optimizer(params)
        self.gamma=0.1
        "_______________________________________________________"
        # 定义记忆库
        #self._meta_conf.num_class=self._meta_conf.statistics["n_classes"],
        self.mem = memory.CSTU(
            capacity=self._meta_conf.memory_size, 
            num_class=self._meta_conf.statistics["n_classes"], 
            lambda_t=self._meta_conf.lambda_t, 
            lambda_u=self._meta_conf.lambda_u
            )
        print(self.mem)
        # 定义EMA模型
        self.model_ema = self.build_ema(self._model)
        # 定义图像增强方式
        self.transform = get_tta_transforms(self._meta_conf)
        # 定义EMA模型更新速率
        self.nu = self._meta_conf.nu
        # 定义模型更新频率
        self.update_frequency = self._meta_conf.update_frequency  # actually the same as the size of memory bank
        self.fishers = None
        # 记录当前训练到的样本数
        self.current_instance = 0

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    
    def one_adapt_step(#forward_and_adapt
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        #adapt the model in one step.
        # batch data
        # 获取伪标签
        with timer("forward"):
            with torch.no_grad():
                #important
                model.eval()#adapt_and_eval中有该函数
                self.model_ema.eval()
                with fork_rng_with_seed(random_seed):
                    ema_out = self.model_ema(batch._x)
                    predict = torch.softmax(ema_out, dim=1)
                    pseudo_label = torch.argmax(predict, dim=1)
                    entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        # 将样本添加到memory bank
        for i, data in enumerate(batch._x):
            #print("!!!!!!i=",i)
            #print("!!!!i=",i)
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            # 每隔一定步数更新模型
           #print("current_instance:", self.current_instance)
            #print("update_frequency:", self.update_frequency)
            if self.current_instance % self.update_frequency == 0:
                #with timer("backward"):
                    model.train()
                    self.model_ema.train()
                    # get memory data
                    # 从记忆库中获取数据
                    sup_data, ages = self.mem.get_memory()
                    l_sup = None
                    if len(sup_data) > 0:
                       # print("88888888888")
                        sup_data = torch.stack(sup_data)
                        strong_sup_aug = self.transform(sup_data)
                        ema_sup_out = self.model_ema(sup_data)
                        stu_sup_out = model(strong_sup_aug)
                        instance_weight = timeliness_reweighting(ages,self._meta_conf.memory_size)
                        l_sup = (adaptation_utils.teacher_student_softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

                    # 优化模型
                    loss = l_sup
                    if loss is not None:
                       #print("999999999")
                        with timer("backward"):
                            optimizer.zero_grad()
                            loss.backward()
                            grads = dict(
                                (name, param.grad.clone().detach())
                                for name, param in model.named_parameters()
                                if param.grad is not None
                            )
                            optimizer.step()
                            #optimizer.zero_grad() 
                        # 更新EMA模型
                        self.model_ema=self.update_ema_variables(
                            self.model_ema, 
                            self._model, 
                            self.nu)
                #self.update_model(model, optimizer)
                #with与return对齐
                        return {
                            "optimizer": copy.deepcopy(optimizer).state_dict(),
                            "loss": loss.item(),
                            "grads": grads,
                            "yhat": ema_out,
                        }
        #with timer("backward"):
        model.train()
        self.model_ema.train()
        # get memory data
        # 从记忆库中获取数据
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
                #print(888888)
                sup_data = torch.stack(sup_data)
                strong_sup_aug = self.transform(sup_data)
                ema_sup_out = self.model_ema(sup_data)
                stu_sup_out = model(strong_sup_aug)
                instance_weight = timeliness_reweighting(ages,self._meta_conf.memory_size)
                l_sup = (adaptation_utils.teacher_student_softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

            # 优化模型
        loss = l_sup
        if loss is not None:
                #print(99999)
            with timer("backward"):    
                optimizer.zero_grad()
                loss.backward()
                grads = dict(
                    (name, param.grad.clone().detach())
                    for name, param in model.named_parameters()
                    if param.grad is not None
                )
                optimizer.step()
                #optimizer.zero_grad()  
            # 更新EMA模型
            #self.model_ema=
            self.update_ema_variables(
                self.model_ema, 
                self._model, 
                self.nu)
            return {
                "optimizer": copy.deepcopy(optimizer).state_dict(),
                "loss": loss.item(),
                "grads": grads,
                "yhat": ema_out,
            }
####################################################################################################    
    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            #print("!!!!!!!!!!!!nbsteps:",nbsteps)
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                ) 
    


    @property
    def name(self):
        return "rotta"

def timeliness_reweighting(ages,memory_size):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float()#.cuda()
    #多/64    
    return torch.exp(-ages/memory_size) / (1 + torch.exp(-ages/memory_size))   
def get_tta_transforms(cfg, gaussian_std: float=0.005, soft=False):
    img_shape = (*cfg.input_size, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, gaussian_std),
        Clip(clip_min, clip_max)
    ])
    return tta_transforms
   
class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + "(min_val={0}, max_val={1})".format(
            self.min_val, self.max_val
        )

class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(
                    1e-8, 1.0
                )  # to fix Nan values in gradients, which happens when applying gamma
                # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        format_string += ", gamma={0})".format(self.gamma)
        return format_string

    """def one_adapt_step(#forward_and_adapt
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        #adapt the model in one step.
        # batch data
        # 获取伪标签
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch._x)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        # 将样本添加到memory bank
        for i, data in enumerate(batch._x):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            # 每隔一定步数更新模型
            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return ema_out"""

    """def update_model(self, model, optimizer):
        # 训练模型和EMA模型
        model.train()
        self.model_ema.train()
        # get memory data
        # 从记忆库中获取数据
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (adaptation_utils.teacher_student_softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        # 优化模型
        l = l_sup
        if l is not None:
            l.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.zero_grad()
            optimizer.step() 
        # 更新EMA模型
        self.update_ema_variables(self.model_ema, self._model, self.nu)"""