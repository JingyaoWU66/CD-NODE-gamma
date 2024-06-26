import torch
import torch.nn as nn
import numpy as np
from . import odeint
from .misc import _flatten, _flatten_convert_none_to_zeros

class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 8, 'Internal error: all arguments required.'
        #y0, func, t, flat_params, rtol, atol, method, options = \
        #    args[:-7], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]

        y0, func, t, xa, flat_params, rtol, atol, method, options = \
            args[:-8], args[-8], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]

        ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options = func, rtol, atol, method, options

        with torch.no_grad():
            ans = odeint(func, y0, t, xa, rtol=rtol, atol=atol, method=method, options=options)

            ###################### adding features to adjoint saved variables ###########

            #if len(y0) == 1:
            #    ans = (torch.cat((ans[0],xa),2),)

        ctx.save_for_backward(t, flat_params, *ans, xa)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        #print('aaaa')
        #torch.autograd.set_detect_anomaly(True) #add by Jennifer
        t, flat_params, *ans, xa = ctx.saved_tensors
        ans = tuple(ans)
        func, rtol, atol, method, options = ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options
        n_tensors = len(ans)
        f_params = tuple(func.parameters())

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                func_eval = func(t, y)
                #print(y[0].shape)

                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                ) # computing the jacabian vectors of f/theta or f/y

            ##################### added by sally
            #vjp_y = vjp_y_and_params[:n_tensors][0][:,:1] # only taking the first column
            #######################################
            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*func_eval, *vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []

            for i in range(T - 1, 0, -1):
                ans_i = tuple(ans_[i] for ans_ in ans) #features taken from each frame
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)

                ############################## added by sally ########################
                func_i = func(t[i],(torch.cat((ans_i[0],xa[i,:,:]),1),))
                #func_i = func(t[i], ans_i)

                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)#[:,:2]
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])

                ####################### added by Sally ##############
                ####################### This section aims to incorprate features #####
                #all_fea = torch.flip(ans[0][i-1:i+1, :, 1:],[0]) # only past the two features
                #ans_i = [all_fea,ans_i[0]]

                #adj_y = (adj_y[0][:, :1],)
                aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)
                #print('iteration number = ' + str(i))
                aug_ans = odeint(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]), xa[i-1,:,:], rtol=rtol, atol=atol, method=method, options=options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[n_tensors:2 * n_tensors]
                adj_time = aug_ans[2 * n_tensors]
                adj_params = aug_ans[2 * n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            ########################## test ##################
            #adj_y=torch.unsqueeze(adj_y[0],0).repeat(100,1,1) #repeating 100 times
            #adj_y =torch.reshape(adj_y,(adj_y.shape[0] * adj_y.shape[1],adj_y.shape[2]))

            #adj_y=(adj_y,)
            ########################## test ##################

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            return (*adj_y, None, time_vjps,None, adj_params, None, None, None, None, None)


def odeint_adjoint(func, y0, t, xa, rtol=1e-6, atol=1e-12, method=None, options=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    if torch.is_tensor(y0):

        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                #print('mmmmmm')
                return (self.base_func(t, y[0]),) #add y0 Jennifer

        tensor_input = True
        #y0 = (y0[0,:,:],)
        y0 = (y0,)
        func = TupleFunc(func)

        ###########################added ny sally ############
        xa = (xa,)

    flat_params = _flatten(func.parameters())
    ys = OdeintAdjointMethod.apply(*y0, func, t, *xa, flat_params, rtol, atol, method, options)

    if tensor_input:
        ys = ys[0]
    return ys
