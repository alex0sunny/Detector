import numpy as np


class Butter(object):
    def __init__(self, btype="lowpass", cutoff=None,
                 cutoff1=None, cutoff2=None,
                 rolloff=48, sampling=None):
        # input checking
        valid = []
        for k, v in zip(["cutoff", "cutoff1", "cutoff2", "rolloff", "sampling"],
                        [cutoff, cutoff1, cutoff2, rolloff, sampling]):
            if type(v) in [int, float]:
                valid.append(k)
        print(str(valid))

        valid = map(lambda k: k[0],
                    filter(lambda k: type(k[1]) in [int, float],
                           zip(["cutoff", "cutoff1", "cutoff2", "rolloff", "sampling"],
                               [cutoff, cutoff1, cutoff2, rolloff, sampling])
                           )
                    )
        if None in [rolloff, sampling]:
            raise ValueError(
                "Butter:rolloff and sampling required for %s filter" % btype)
        if "rolloff" not in valid:
            raise TypeError("Butter:invalid rolloff argument")
        if "sampling" not in valid:
            raise TypeError("Butter:invalid sampling argument")
        if btype in ["lowpass", "highpass", "notch"]:
            if None in [cutoff]:
                raise ValueError(
                    "Butter:cutoff required for %s filter" % btype)
            if "cutoff" not in valid:
                raise TypeError("Butter:invalid cutoff argument")
        elif btype in ["bandpass", "bandstop"]:
            if None in [cutoff1, cutoff2]:
                raise ValueError(
                    "Butter:cutoff1 and cutoff2 required for %s filter" % btype)
            if "cutoff1" not in valid:
                raise TypeError("Butter:invalid cutoff1 argument")
            if "cutoff2" not in valid:
                raise TypeError("Butter:invalid cutoff2 argument")
            if cutoff1 > cutoff2:
                raise ValueError(
                    "Butter:cutoff1 must be less than or equal to cutoff2")
        else:
            raise ValueError("Butter: invalid btype %s" % btype)


filter_ = Butter(btype='bandpass', cutoff1=float(0.1), cutoff2=float(1.0), sampling=1000)

