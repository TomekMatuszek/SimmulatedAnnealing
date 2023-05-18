from algorithms import LocationAnnealing
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

def parallel_annealing(data, iter, init='highest', move_choice='random', neighbourhood=2, n_shops=6, buffer=1500, objective=250000, start_temp=20000, temp_mult=None, temp_substr=None):
    la = LocationAnnealing(data)
    res = la.run(init, move_choice, neighbourhood, n_shops, buffer, objective, start_temp, temp_mult, temp_substr)
    return (iter, res, (init, move_choice, neighbourhood, n_shops, buffer, objective, start_temp, temp_mult, temp_substr))