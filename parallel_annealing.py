from algorithms import LocationAnnealing
import warnings

def parallel_annealing(data, resolution, iter, init='highest', move_choice='random', neighbourhood=2, n_shops=6, buffer=1500,
                       objective=None, n_evals=None, prop_rejected=None, start_temp=0.001, temp_mult=None, temp_substr=None):
    warnings.simplefilter(action='ignore', category=UserWarning)
    la = LocationAnnealing(data, resolution)
    res, status = la.run(init, move_choice, neighbourhood, n_shops, buffer, objective, n_evals, prop_rejected, start_temp, temp_mult, temp_substr)
    return (iter, res, (init, move_choice, neighbourhood, n_shops, buffer, objective, n_evals, prop_rejected, start_temp, temp_mult, temp_substr))