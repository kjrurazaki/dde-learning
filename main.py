from modules.dde import DDE
from matplotlib import pyplot as plt
import numpy as np


def dde_expression(x, x_delayed, param):
    """
    : params
    :: x is the current value of variable
    :: x_delayed is the value of variable in the applied delay
    """
    dx_dt = param[0] * x * (1 - x_delayed)
    return dx_dt



def main():
    dde_logistics = DDE(var = np.array([1]), 
                        hist_value = 0.5, 
                        delay = 1,
                        exp = dde_expression,
                        param = np.array([np.pi / 2]))
    print(dde_logistics.eval_dx_dt())
    dde_logistics.solution_in_time(timestep=0.1, 
                                   t_final=10)
    plt.plot(dde_logistics.t, dde_logistics.vars[:, 0],
                                   label = "0.1")
    dde_logistics.solution_in_time(timestep=0.0001, 
                                   t_final=10)
    plt.plot(dde_logistics.t, dde_logistics.vars[:, 0],
                                   label = "0.0001")
    plt.legend()
    plt.show()
    print("debug")


if __name__ == "__main__":
    main()
