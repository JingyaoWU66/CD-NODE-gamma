# Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
import torch
from .misc import (
    _scaled_dot_product, _convert_to_tensor, _is_finite, _select_initial_step, _handle_unused_kwargs, _is_iterable,
    _optimal_step_size, _compute_error_ratio
)
from .solvers import AdaptiveStepsizeODESolver
from .interp import _interp_fit, _interp_evaluate
from .rk_common import _RungeKuttaState, _ButcherTableau, _runge_kutta_step, rk4_step_func, rk4_alt_step_func

_DORMAND_PRINCE_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.],
    beta=[
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    c_sol=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    c_error=[
        35 / 384 - 1951 / 21600,
        0,
        500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720,
        -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300,
        -1. / 60.,
    ],
)

DPS_C_MID = [
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
    187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
]


def _interp_fit_dopri5(y0, y1, k, dt, tableau=_DORMAND_PRINCE_SHAMPINE_TABLEAU):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    dt = dt.type_as(y0[0])

    ####################### added by sally #################
    #print("gg")
    #print(len(y0))
    if len(y0) == 1:
        y_mid = tuple(y0_[:,:1] + _scaled_dot_product(dt, DPS_C_MID, k_) for y0_, k_ in zip(y0, k)) # y0 has been changed
        ##print("g")
        #print(y_mid)
    else:
        y_mid = tuple(
            y0_ + _scaled_dot_product(dt, DPS_C_MID, k_) for y0_, k_ in zip(y0, k))  # y0 has been changed
        #print(y_mid)

    f0 = tuple(k_[0] for k_ in k)
    f1 = tuple(k_[-1] for k_ in k)
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _abs_square(x):
    return torch.mul(x, x)


def _ta_append(list_of_tensors, value):
    """Append a value to the end of a list of PyTorch tensors."""
    list_of_tensors.append(value)
    return list_of_tensors


class Dopri5Solver(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, xa, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.xa = xa
        #self.xa = xa[:,:,:-1] # add by jennifer
        #print('hello')
        #print(xa.shape)
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)

    def before_integrate(self, t):
        ##################### added by sally ######################
       # print("gg")
        #print(len(self.y0))
        #print(self.y0)
        if len(self.y0)==1:
            f0 = self.func(t[0].type_as(self.y0[0]), (torch.cat((self.y0[0], self.xa[0,:,:]),1),))
        else:
            self.y0_1 = (torch.cat((self.y0[0], self.xa), 1),)  + self.y0[1:]
            f0 = self.func(t[0].type_as(self.y0[0]), self.y0_1)
            f0 = (f0[0],) + (f0[1][:,:f0[0].shape[-1]],) + (f0[2],) + (f0[3],)

        ############################################################

        if self.first_step is None:
            #first_step = _select_initial_step(self.func, t[0], self.y0, 4, self.rtol[0], self.atol[0], f0=f0).to(t)
            if len(self.y0)==1:
                first_step = _select_initial_step(self.func, t[0], (torch.cat((self.y0[0], self.xa[0,:,:]),1),),self.xa, 4, self.rtol[0], self.atol[0], f0=f0).to(t)
            else:
                first_step = _select_initial_step(self.func, t[0], self.y0,
                                                  self.xa, 4, self.rtol[0], self.atol[0], f0=f0).to(t)
        else:
            first_step = _convert_to_tensor(0.01, dtype=t.dtype, device=t.device)


        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, interp_coeff=[self.y0] * 5)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        if len(self.y0)==1:
            ind = (next_t / 0.040 - 1).int()
            #print('current frame number is ' + str(ind))
            x0 = self.xa[ind + 1,:, :] # only take the original features; fixed
        else:
            x0 = self.xa
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_dopri5_step(self.rk_state, x0)
            #print(self.rk_state.y1[0].shape)
            n_steps += 1

        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)



    def _adaptive_dopri5_step(self, rk_state, x0):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        #print('------------000000hello')
        #print('------hhhhhhhhh')
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        #print('------------1111111hello')
        #####################################################################################
        #####################################################################################
        #####################################################################################
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, x0, f0, t0, dt, tableau=_DORMAND_PRINCE_SHAMPINE_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = (torch.tensor(mean_sq_error_ratio) <= 1).all()

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0


        #interp_coeff = _interp_fit_dopri5(y0, y1, k, dt) if accept_step else interp_coeff

        ########################added by sally
        interp_coeff = _interp_fit_dopri5(y0, y1, k, dt) if accept_step else interp_coeff
        #####################################

        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=5
        )

        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state
