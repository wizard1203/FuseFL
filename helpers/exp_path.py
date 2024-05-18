import os, sys
import logging

import datetime
import pickle
from copy import deepcopy

# from .meter import AverageMeter
from .wandb_utils import wandb_tool
from .metrics_sync import Metrics_Sync





class ExpTool:
    TIMEFORMAT = '%Y%m%d_%H%M%S'
    theTime = datetime.datetime.now()
    begin_time = theTime.strftime(TIMEFORMAT) + theTime.strftime('%f')[:2]

    # root_dir_name = f"wandb_summary_{wandb.run.project}"
    root_dir_name = f"exp_savefiles"

    exp_name = None
    wandb_open = False
    # role = None

    history = []
    _step = 0
    summary_dict = Metrics_Sync.summary_dict



    @classmethod
    def logging_write(cls, string, fd):
        logging.info(string)
        fd.write(string + '\n')



    @classmethod
    def init(cls, args):
        cls.config = args
        cls.wandb_open = args.enable_wandb

        cls.project_name = args.project_name
        cls.exp_name = args.exp_name
        cls.root_dir_name = os.path.join(args.exp_abs_path, f"exp_savefiles_{cls.project_name}")
        logging.info(f"EXP path init, abs path:{os.path.abspath(cls.root_dir_name)}")

        cls.run_name = f"{cls.exp_name}_{cls.begin_time}"
        cls.sub_dir_name = f"{cls.root_dir_name}/{cls.run_name}"

        cls.make_dir()
        # wandb_save_dir = cls.sub_dir_name
        # wandb_tool.init(args, wandb_save_dir)
        wandb_tool.init(args)


    @classmethod
    def init_with_sub_dir(cls, args, init_with_sub_dir):
        cls.config = args
        cls.wandb_open = args.enable_wandb

        cls.project_name = args.project_name
        cls.exp_name = args.exp_name
        cls.root_dir_name = os.path.join(args.exp_abs_path, f"exp_savefiles_{cls.project_name}")
        logging.info(f"EXP path init, under subdir path:{init_with_sub_dir}")

        cls.run_name = f"{cls.exp_name}_{cls.begin_time}"
        cls.sub_dir_name = f"{init_with_sub_dir}"

        cls.make_dir()
        # wandb_save_dir = cls.sub_dir_name
        # wandb_tool.init(args, wandb_save_dir)
        wandb_tool.init(args)


    @classmethod
    def get_sub_dir_name(cls):
        return cls.sub_dir_name

    @classmethod
    def get_file_name(cls, file_name, exp_dir=True):
        if exp_dir:
            return os.path.join(cls.sub_dir_name, file_name)
        else:
            return file_name



    @classmethod
    def check_file_exist(cls, file_name, exp_dir=True):
        if exp_dir:
            return os.path.exists(os.path.join(cls.sub_dir_name, file_name))
        else:
            return os.path.exists(file_name)

    @classmethod
    def check_pickle_exist(cls, file_name, exp_dir=True):
        if exp_dir:
            return os.path.exists(f'{cls.sub_dir_name}/{file_name}.pickle')
        else:
            return os.path.exists(f'{file_name}.pickle')

    @classmethod
    def save_pickle(cls, data, file_name, exp_dir=True):
        if exp_dir:
            with open(f'{cls.sub_dir_name}/{file_name}.pickle', 'wb') as file:
                pickle.dump(data, file)
        else:
            with open(f'{file_name}.pickle', 'wb') as file:
                pickle.dump(data, file)

    @classmethod
    def load_pickle(cls, file_name, exp_dir=True):
        if exp_dir:
            with open(f'{cls.sub_dir_name}/{file_name}.pickle', 'rb') as file:
                data = pickle.load(file)
        else:
            with open(f'{file_name}.pickle', 'rb') as file:
                data = pickle.load(file)
        return data

    @classmethod
    def make_dir(cls):
        if not os.path.exists(cls.root_dir_name):
            os.makedirs(cls.root_dir_name)
        if not os.path.exists(cls.sub_dir_name):
            os.makedirs(cls.sub_dir_name)


    @classmethod
    def record(cls, new_values, update_summary=True):
        """
            prefix + tags.values is the name of sp_values;
            values should include information like:
            {"Acc": 0.9, "Loss":}
            com_values should include information like:
            {"epoch": epoch, }
        """
        if cls.wandb_open:
            wandb_tool.record(new_values=new_values, update_summary=update_summary)
        Metrics_Sync.record(new_values, update_summary)


    @classmethod
    def merge_from_sub_history(cls, sub_history):
        cls.history += sub_history
        for values in sub_history:
            Metrics_Sync.record(values, update_summary=True)


    @classmethod
    def summary(cls, values):
        if cls.wandb_open:
            wandb_tool.summary(values)
        Metrics_Sync.summary(values)

    @classmethod
    def get_sub_history(cls, reset=True):
        copied_sub_history = deepcopy(cls.history)
        if reset:
            cls.history = []
        return copied_sub_history

    @classmethod
    def upload(cls, reset=True):
        if cls.wandb_open:
            wandb_tool.upload(reset)
        cls.history.append(dict(
            _step=cls._step,
            **Metrics_Sync.round_sync_dict
        ))
        cls._step += 1
        Metrics_Sync.reset()


    @classmethod
    def finish(cls, config):
        if cls.wandb_open:
            wandb_run_path, wandb_exp_name = wandb_tool.finish(config)
            logging.info(f"Uploading exp results to wandb, wandb_run_path:{wandb_run_path}, wandb_exp_name:{wandb_exp_name}")
        else:
            wandb_run_path, wandb_exp_name = None, None
        TIMEFORMAT = '%Y%m%d_%H%M%S'
        theTime = datetime.datetime.now()
        end_time = theTime.strftime(TIMEFORMAT) + theTime.strftime('%f')[:2]
        run_name = cls.exp_name
        file_name = f"{cls.sub_dir_name}/exp_summary.log"
        logging.info(f"Saving exp results file at {file_name}")
        with open(file_name, "a+") as fo:
            fo.write(f'===================Begin Run======================================\n')
            # fo.write(f'wandb.run.id: {wandb.run.id}\n')
            # fo.write(f'wandb.run.path: {wandb.run.path}\n')
            fo.write(f'exp name: {run_name}\n')
            fo.write(f'Begin Time: {cls.begin_time}\n')
            fo.write(f'End Time: {end_time}\n')
            for key, val in dict(cls.summary_dict).items():
                fo.write(f'{key}: {val}\n')
            fo.write(f'===================End Run======================================\n')
            fo.write(f'wandb_run_path: {wandb_run_path} \n')
            fo.write(f'wandb_exp_name: {wandb_exp_name} \n')
        with open(f'{cls.sub_dir_name}/history.pickle', 'wb') as file:
            pickle.dump(cls.history, file)
        with open(f'{cls.sub_dir_name}/summary.pickle', 'wb') as file:
            pickle.dump(cls.summary_dict, file)






