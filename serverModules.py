from shiny import module, render, ui, _namespaces, reactive, Session
from algorithms import LocationAnnealing, LocationEvolution, Parameters
from parallel_annealing import parallel_annealing
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing as mp
import re
import warnings

@module.server
def SA_server(input, output, session:Session):
    result:str = reactive.Value("")
    SAlocations:LocationAnnealing = reactive.Value(None)

    @output
    @render.ui
    def term_obj():
        if input.objective():
            return ui.TagList(ui.input_numeric("pop_objective", "Population objective", value=250000))
        else:
            return None
    
    @output
    @render.ui
    def term_eval():
        if input.evals():
            return ui.TagList(ui.input_numeric("n_evals", "Number of evaluations", value=500))
        else:
            return None
    
    @output
    @render.ui
    def term_prop():
        if input.prop() or (not input.objective() and not input.evals()):
            ui.update_checkbox('prop', value=True)
            return ui.TagList(ui.input_slider("n_rejected", "Proportion of rejected permutations", 0, 1, value=0.95))
        else:
            return None

    @output
    @render.ui
    def mult_substr():
        if input.temp_change() == 'multiply':
            return ui.TagList(
                ui.input_slider("temp_mult", "Temperature multiplier", 0, 1, 0.95)
            )
        elif input.temp_change() == 'substract':
            return ui.TagList(
                ui.input_numeric("temp_substr", "Temperature substractor", value=100)
            )

    @output
    @render.plot(alt="Map")
    @reactive.event(input.run)
    def map():
        warnings.simplefilter(action='ignore', category=UserWarning)
        file = input.grid_file()
        if file is not None:
            data:gpd.GeoDataFrame = gpd.read_file(file[0]["datapath"])
            resolution = int(re.findall('[0-9]+', file[0]["name"])[0])
            la = LocationAnnealing(data, resolution)

            if input.temp_change() == 'multiply':
                temp_mult = float(input.temp_mult())
                temp_substr = None
            elif input.temp_change() == 'substract':
                temp_substr = int(input.temp_substr())
                temp_mult = None

            if input.objective():
                objective = int(input.pop_objective())
            else:
                objective = None
            if input.evals():
                n_evals = int(input.n_evals())
            else:
                n_evals = None
            if input.prop():
                prop_rejected = float(input.n_rejected())
            else:
                prop_rejected = 0.999
            
            params = Parameters(
                init=str(input.init()),
                move_choice=str(input.movement()),
                neighbourhood=int(input.neighbourhood()),
                n_shops=int(input.n_shops()),
                buffer=int(input.buffer()),
                objective=objective,
                n_evals=n_evals,
                prop_rejected=prop_rejected,
                start_temp=int(input.init_temp()),
                temp_mult=temp_mult,
                temp_substr=temp_substr
            )

            ui.notification_show('Annealing...', duration=1000, id='message')
            res, evals, status = la.run(params)
            ui.notification_remove('message')
            SAlocations.set(res)
            match status:
                case 1:
                    ui.notification_show(f'Population objective achieved after {evals-1} iterations', duration=10, id='status')
                case 2:
                    ui.notification_show(f'{evals-1} iterations performed', duration=10, id='status')
                case 3:
                    ui.notification_show(f'Exceeded proportion of rejected permutations after {evals-1} iterations', duration=10, id='status')
            proc = round(la.objective / la.grid['ludnosc'].sum() * 100, 2)
            result.set(f'Population covered: {round(la.objective)} ({proc}%)')
            fig, ax = la.plot_map()
            fig.tight_layout()
    
    @output
    @render.plot(alt="Trend")
    def temp_trend():
        init_temp = input.init_temp()
        threshold = 0.005 * init_temp
        values = []
        while init_temp > threshold:
            values.append(init_temp)
            if input.temp_change() == 'multiply':
                temp_mult = input.temp_mult()
                init_temp *= temp_mult
            elif input.temp_change() == 'substract':
                temp_substr = input.temp_substr()
                init_temp -= temp_substr
        
        plt.plot(values, color='red')
        plt.title('Temperature lowering trend')

    @output
    @render.plot(alt="Objective plot")
    def obj_plot():
        result = SAlocations()
        if result is not None:
            result.plot_objective()
    
    @output
    @render.plot(alt="Probability plot")
    def prob_plot():
        result = SAlocations()
        if result is not None:
            result.plot_probs()
    
    @output
    @render.text
    def result_objective():
        return str(result())
    
    @session.download(filename="results.csv")
    def download_results():
        res:gpd.GeoDataFrame = SAlocations()
        yield 'id;X;Y\n'
        if res is not None:
            ids = res.index
            x_coords = res.geometry.centroid.x
            y_coords = res.geometry.centroid.y
            for id, x, y in zip(ids, x_coords, y_coords):
                yield f'{id};{x};{y}\n'

@module.server
def GA_server(input, output, session):
    resultSA:str = reactive.Value("")
    resultGA:str = reactive.Value("")
    GAlocations:gpd.GeoDataFrame = reactive.Value(None)

    @output
    @render.ui
    def term_obj():
        if input.objective():
            return ui.TagList(ui.input_numeric("pop_objective", "Population objective", value=250000))
        else:
            return None
    
    @output
    @render.ui
    def term_eval():
        if input.evals():
            return ui.TagList(ui.input_numeric("n_evals", "Number of evaluations", value=500))
        else:
            return None
    
    @output
    @render.ui
    def term_prop():
        if input.prop() or (not input.objective() and not input.evals()):
            ui.update_checkbox('prop', value=True)
            return ui.TagList(ui.input_slider("n_rejected", "Proportion of rejected permutations", 0, 1, value=0.95))
        else:
            return None

    @output
    @render.ui
    def mult_substr():
        if input.temp_change() == 'multiply':
            return ui.TagList(
                ui.input_slider("temp_mult", "Temperature multiplier", 0, 1, 0.95)
            )
        elif input.temp_change() == 'substract':
            return ui.TagList(
                ui.input_numeric("temp_substr", "Temperature substractor", value=100)
            )

    @output
    @render.plot(alt="Map")
    @reactive.event(input.run)
    def map():
        warnings.simplefilter(action='ignore', category=UserWarning)
        file = input.grid_file()
        if file is not None:
            data:gpd.GeoDataFrame = gpd.read_file(file[0]["datapath"])
            resolution = int(re.findall('[0-9]+', file[0]["name"])[0])
            if input.temp_change() == 'multiply':
                temp_mult = float(input.temp_mult())
                temp_substr = None
            elif input.temp_change() == 'substract':
                temp_substr = int(input.temp_substr())
                temp_mult = None
          
            if input.objective():
                objective = int(input.pop_objective())
            else:
                objective = None
            if input.evals():
                n_evals = int(input.n_evals())
            else:
                n_evals = None
            if input.prop():
                prop_rejected = float(input.n_rejected())
            else:
                prop_rejected = 0.95
            
            with ui.Progress(0, 100) as p:
                warnings.simplefilter(action='ignore', category=UserWarning)
                p.set(5, 'Preparing...')
                pool = mp.Pool(input.workers())
                params = Parameters(
                    str(input.init()), str(input.movement()), int(input.neighbourhood()), int(input.n_shops()),
                    int(input.buffer()), objective, n_evals, prop_rejected,
                    int(input.init_temp()), temp_mult, temp_substr
                )
                p.set(10, 'Annealing...')
                
                result_objects = [pool.apply_async(parallel_annealing, args=(data, resolution, i, params)) for i in range(0, input.iterations())]
                results = [r.get() for r in result_objects]
                pool.close()
                pool.join()
                p.set(50, 'Processing...')

                results.sort(key = lambda x: x[0])
                results = [r for i, r, params in results]
                resultSA.set(f'Best score from SA: {round(results[0].objective)}')
                p.set(60, 'Evolution...')
                population = [res.shops for res in results]
                le = LocationEvolution(data, resolution, population, results[0].buffer)
                res = le.run(input.epochs())
                p.set(95, 'Plotting...')
            proc = round(max(le.scores) / le.grid['ludnosc'].sum() * 100, 2)
            resultGA.set(f'Best score after GA: {round(max(le.scores))} ({proc}%)')
            GAlocations.set(res.shops)
            le.plot_map()

    @output
    @render.plot(alt="Trend")
    def temp_trend():
        init_temp = input.init_temp()
        threshold = 0.005 * init_temp
        values = []
        while init_temp > threshold:
            values.append(init_temp)
            if input.temp_change() == 'multiply':
                temp_mult = input.temp_mult()
                init_temp *= temp_mult
            elif input.temp_change() == 'substract':
                temp_substr = input.temp_substr()
                init_temp -= temp_substr
        
        plt.plot(values, color='red')
        plt.title('Temperature lowering trend')
    
    @output
    @render.text
    def result_objective_start():
        return str(resultSA())
    
    @output
    @render.text
    def result_objective_end():
        return str(resultGA())
    
    @session.download(filename="results.csv")
    def download_results():
        res:gpd.GeoDataFrame = GAlocations()
        yield 'id;X;Y\n'
        if res is not None:
            ids = res.index
            x_coords = res.geometry.centroid.x
            y_coords = res.geometry.centroid.y
            for id, x, y in zip(ids, x_coords, y_coords):
                yield f'{id};{x};{y}\n'
