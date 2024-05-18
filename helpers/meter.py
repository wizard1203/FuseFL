from copy import deepcopy

class MaxMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.max = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max


class MinMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.min = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.min is None or value < self.min:
            self.min = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.min

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

    def make_summary(self, key="None"):
        avg_key = key + "/" + "avg"
        max_key = key + "/" + "max"
        final_key = key + "/" + "final"
        summary = {
            avg_key: self.avg,
            max_key: self.max,
            final_key: self.val,
        }
        return summary

    def get_summary(self):
        return self.val, self.avg, self.sum, self.max, self.min, self.count

    # def make_summary(self, key="None"):
    #     sum_key = key + "/" + "sum"
    #     count_key = key + "/" + "count"
    #     avg_key = key + "/" + "avg"
    #     max_key = key + "/" + "max"
    #     min_key = key + "/" + "min"
    #     final_key = key + "/" + "final"
    #     summary = {
    #         sum_key: self.sum,
    #         count_key: self.count,
    #         avg_key: self.avg,
    #         max_key: self.max,
    #         min_key: self.min,
    #         final_key: self.val,
    #     }
    #     return summary

class HistoryMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

    def make_summary(self, key="None"):
        avg_key = key + "/" + "avg"
        max_key = key + "/" + "max"
        final_key = key + "/" + "final"
        summary = {
            avg_key: self.avg,
            max_key: self.max,
            final_key: self.val,
        }
        return summary

    def get_summary(self):
        return self.val, self.avg, self.sum, self.max, self.min, self.count




def moving_avg(old_value, new_value, iteration, factor=0.8):
    if iteration == 0:
        EMA_value = new_value
    else:
        EMA_value = old_value * factor + new_value * (1 - factor)
    return EMA_value




class StreamAverager(object):
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        # moving_avg will not be reset
        self.moving_avg = None
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.count = 0


    def update(self, avg, n=1):
        self.sum += avg * n
        self.count += n
        self.avg = self.sum / self.count
        if self.moving_avg is None:
            self.moving_avg = avg
        else:
            self.moving_avg = (1 - self.alpha) * self.moving_avg + self.alpha * self.avg


    def get_sub_info_to_agg(self):
        return {"avg": self.avg, "n": self.count}

    def to_dict(self):
        return {"avg": self.avg, "sum": self.sum, "count": self.count}

    def computer(self):
        return self.sum

    def make_summary(self, key="None"):
        avg_key = key + "/" + "avg"
        max_key = key + "/" + "max"
        summary = {
            avg_key: self.avg,
            max_key: self.sum,
        }
        return summary



class StreamList_recorder(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.list = []
        self.count = 0

    # @property

    def update(self, vals):
        if isinstance(vals, list):
            self.list += vals
            self.count += len(vals)

        else:
            self.list.append(vals)

        self.count += n
        self.avg = self.sum / self.count

    def computer(self):
        return self.sum

    def make_summary(self, key="None"):
        max_key = key + "/" + "max"
        summary = {
            max_key: self.sum,
        }
        return summary


















