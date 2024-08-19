import numpy as np


class DDE:
    """
    Implementation of 1D DDE (dx_dt) with unique delay
    :init params
    :: var: array of variables, should be organized by columns and initial value as first line
    :: hist_value: value taken by the function for t<0
    :: delay: delay of the equation
    :: exp: dde expression (as fuction)
    :: param: parameters of the DDE, if required
    """

    def __init__(self, var, hist_value, delay, exp, param=None):
        self.param = param
        self.exp = exp
        self.hist_value = hist_value
        self.delay = delay
        self.t = [0]
        self.vars = np.column_stack([var, np.array([hist_value])])

    def eval_dx_dt(self):
        return self.exp(*self.vars[-1,:], self.param)

    def update_vars(self, new_values):
        self.vars = np.vstack([self.vars, 
                               np.array(new_values)])

    def update_t(self, new_t):
        self.t = np.append(self.t, new_t)

    def get_var_value(self, t):
        """
        Return value of variable at t (interpolated)
        """
        return np.interp(t, self.t, self.vars[:, 0])

    def history(self, t):
        """
        Define function for history
        """
        return self.hist_value if t < 0 else self.get_var_value(t)

    def solution_in_time(self, timestep, t_final):
        """
        Simple forward Euler to solve DDE
        """
        while self.t[-1] <= t_final:
            # Derivative approximation
            dx_t = self.eval_dx_dt() * timestep
            propagated_x = self.vars[-1, 0] + dx_t
            t = self.t[-1] + timestep
            # Update x
            x_delayed = self.history(t - self.delay)
            self.update_t(t)
            self.update_vars([propagated_x, x_delayed])
