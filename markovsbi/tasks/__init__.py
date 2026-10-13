from markovsbi.tasks.ar1 import AR1
from markovsbi.tasks.double_well import DoubleWell
from markovsbi.tasks.kolmogorov_flow import KolmogorovFlow
from markovsbi.tasks.linear_sde_task import DataHighDimSDE, GeneralSDE, PeriodicSDE
from markovsbi.tasks.lotka_volterra import LotkaVolterra
from markovsbi.tasks.mixture_rw_nd import MixtureRW2D, MixtureRW5D
from markovsbi.tasks.simple_nd import (
    Simple1D,
    Simple1DNonstationary,
    Simple1DStationary,
    Simple2D,
    Simple2DNonstationary,
    Simple2DStationary,
    Simple10DNonstationary,
    Simple10DStationary,
)
from markovsbi.tasks.sir import SIR


def get_task(task_name, **kwargs):
    if task_name.lower() == "simple1d":
        return Simple1D(**kwargs)
    elif task_name.lower() == "simple1dstationary":
        return Simple1DStationary(**kwargs)
    elif task_name.lower() == "simple1dnonstationary":
        return Simple1DNonstationary(**kwargs)
    elif task_name.lower() == "simple2dnonstationary":
        return Simple2DNonstationary(**kwargs)
    elif task_name.lower() == "simple10dnonstationary":
        return Simple10DNonstationary(**kwargs)
    elif task_name.lower() == "mixture_rw_2d":
        return MixtureRW2D(**kwargs)
    elif task_name.lower() == "mixture_rw_5d":
        return MixtureRW5D(**kwargs)
    elif task_name.lower() == "simple2d":
        return Simple2D(**kwargs)
    elif task_name.lower() == "simple2dstationary":
        return Simple2DStationary(**kwargs)
    elif task_name.lower() == "simple10dstationary":
        return Simple10DStationary(**kwargs)
    elif task_name.lower() == "periodic_sde":
        return PeriodicSDE(**kwargs)
    elif task_name.lower() == "general_sde":
        return GeneralSDE(**kwargs)
    elif task_name.lower() == "10d_sde":
        return DataHighDimSDE(**kwargs)
    elif task_name.lower() == "lotka_volterra":
        return LotkaVolterra(**kwargs)
    elif task_name.lower() == "sir":
        return SIR(**kwargs)
    elif task_name.lower() == "double_well":
        return DoubleWell(**kwargs)
    elif task_name.lower() == "kolmogorov_flow":
        return KolmogorovFlow(64)
    elif task_name.lower() == "ar1":
        return AR1(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not found.")
