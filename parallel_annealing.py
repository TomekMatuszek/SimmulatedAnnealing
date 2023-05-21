from algorithms import LocationAnnealing, Parameters
import warnings

def parallel_annealing(data, resolution, iter, params:Parameters):
    warnings.simplefilter(action='ignore', category=UserWarning)
    la = LocationAnnealing(data, resolution)
    res, status = la.run(params)
    return (iter, res, params.values)