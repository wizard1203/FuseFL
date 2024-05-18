import argparse
import os, sys

import datetime

from .meter import AverageMeter


class Metrics_Sync:
    round_sync_dict = {}
    meter_dict = {}
    summary_dict = {}

    @classmethod
    def record(cls, new_values, update_summary=True):
        for key, value in new_values.items():
            cls.round_sync_dict[key] = value
            if update_summary and type(value) is not str:
                if key not in cls.meter_dict:
                    cls.meter_dict[key] = AverageMeter()
                cls.meter_dict[key].update(value, n=1)
                summary = cls.meter_dict[key].make_summary(key)
                for key, valaue in summary.items():
                    cls.summary_dict[key] = valaue


    @classmethod
    def summary(cls, values):
        for key, valaue in values.items():
            cls.summary_dict[key] = valaue



    @classmethod
    def reset(cls):
        cls.round_sync_dict = {}
















