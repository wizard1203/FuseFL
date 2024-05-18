import argparse
import logging
import os, sys
import pickle

import datetime

import wandb

from abc import ABC, abstractmethod

from .meter import AverageMeter


class wandb_tool:

    round_sync_dict = {}
    meter_dict = {}
    enable = False

    @classmethod
    def init(cls, args):
        cls.args = args
        cls.enable = args.enable_wandb
        cls.exp_name = args.exp_name
        if args.wandb_key is not None:
            wandb.login('allow', args.wandb_key)
        if cls.enable:
            wandb_args = {
                "entity": args.wandb_entity,
                "project": args.project_name,
                "config": args,
                "name": args.exp_name,
            }
            if hasattr(args, "wandb_offline") and args.wandb_offline:
                os.environ['WANDB_MODE'] = 'offline'
            else:
                os.environ['WANDB_MODE'] = 'online'
            if hasattr(args, "wandb_console") and not args.wandb_console:
                os.environ['WANDB_CONSOLE'] = 'off'
            wandb.init(**wandb_args)

    @classmethod
    def record(cls, new_values, update_summary=True):
        """
            prefix + tags.values is the name of sp_values;
            values should include information like:
            {"Acc": 0.9, "Loss":}
            com_values should include information like:
            {"epoch": epoch, }
        """
        if cls.enable:
            for key, value in new_values.items():
                wandb_tool.round_sync_dict[key] = value
                if update_summary and type(value) is not str:
                    if key not in wandb_tool.meter_dict:
                        wandb_tool.meter_dict[key] = AverageMeter()
                    wandb_tool.meter_dict[key].update(value, n=1)
                    summary = wandb_tool.meter_dict[key].make_summary(key)
                    for key, valaue in summary.items():
                        wandb.run.summary[key] = valaue

    @classmethod
    def summary(cls, values):
        for key, valaue in values.items():
            wandb.run.summary[key] = valaue

    @classmethod
    def upload(cls, reset=True):
        if cls.enable:
            wandb.log(wandb_tool.round_sync_dict)
            if reset:
                wandb_tool.round_sync_dict = {}

    @classmethod
    def finish(cls, args):
        if cls.enable:
            wandb_path = wandb.run.path
            wandb.finish()
            return wandb_path, cls.exp_name








