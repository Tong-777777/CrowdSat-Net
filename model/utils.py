# this code is based on https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/models/utils.py
"""
Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
All rights reserved.

This source code is under the MIT License.

Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

Last modification: March 18, 2024

__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"
"""

import torch

from typing import Union, Tuple, List, Optional

__all__ = ['load_model', 'count_parameters', 'LossWrapper']

def load_model(model: torch.nn.Module, pth_path: str, strict: bool=False) -> torch.nn.Module:
    """
    :param model: pre-trained model
    :param pth_path: the path to the PTH file
    :return: torch.nn.Module with pre-trained parameters
    """

    map_location = torch.device('cpu')
    if torch.cuda.is_available():
        map_location = torch.device('cuda')

    checkpoint = torch.load(pth_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    return model

def count_parameters(model: torch.nn.Module) -> tuple:
    """
    :param model:
    :return: the total params within the model
    """

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f'TRAINABLE PARAMETERS: {train_params}')
    print(f'TOTAL PARAMETERS: {all_params}')

    return train_params, all_params


class LossWrapper(torch.nn.Module):
    ''' nn.Module wrapper to add loss output to a model '''

    def __init__(
            self,
            model: torch.nn.Module,
            losses: list = None,
            mode: str = 'module'
    ) -> None:
        '''
        Args:
            model (torch.nn.Module): the model module
            losses (list): loss function and its name
            mode (str, optional): output mode, possible values are:
                - 'loss_only', to output the loss dict only,
                - 'preds_only', to output the predictions only,
                - 'both', to output both loss dict and predictions,
                - 'module' (default), to output loss dict only during training (i.e.
                    model.train()) and both output and loss during evaluation (i.e.
                    model.eval()).
                Defaults to 'module'.
        '''

        super().__init__()

        if losses:
            assert isinstance(losses, list), \
                'losses argument must be a list.'

        assert mode in ['loss_only', 'preds_only', 'both', 'module'], \
            'Wrong mode argument, must be \'loss_only\', \'preds_only\', \'both\', or \'module\'.'

        self.model = model
        self.losses = losses
        self.output_mode = mode

    def forward(
            self,
            x: torch.Tensor,
            target: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    ) -> Union[Tuple[torch.Tensor, dict], dict, torch.Tensor]:
        '''
        Args:
            x (torch.Tensor): input of the model
            target (torch.Tensor or list): target used for the loss computation

        Returns:
            Union[Tuple[torch.Tensor, dict], dict, torch.Tensor]
                depends on mode value
        '''
        try:
            output = self.model(x)
        except ValueError:
            output = self.model(x, target)

        output = self.model(x)

        output_used = output
        if isinstance(output, torch.Tensor):
            output_used = [output]
        if isinstance(target, torch.Tensor):
            target = [target]

        output_dict = {}

        if target is not None:
            for i, dic in zip(output_used, self.losses):
                loss_module = dic['loss']
                loss = loss_module(i, target[0])
                output_dict.update({dic['name']: loss})

        if self.output_mode == 'module':
            if self.training:
                if not output_dict:
                    output_dict = output
                return output_dict
            else:
                return output[1], output_dict

        elif self.output_mode == 'loss_only':
            return output_dict

        elif self.output_mode == 'preds_only':
            return output[1]

        elif self.output_mode == 'both':
            return output[1], output_dict