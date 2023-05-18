from shiny import module, render, ui, _namespaces, reactive
from algorithms import LocationAnnealing, LocationEvolution
from parallel_annealing import parallel_annealing
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing as mp
import warnings

@module.server
def SA_server(input, output, session):
    result = reactive.Value("")

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
            la = LocationAnnealing(data)
            init_method = str(input.init())
            init_method = 'highest' if init_method == 'highest population cells' else 'random'
            if input.temp_change() == 'multiply':
                temp_mult = float(input.temp_mult())
                temp_substr = None
            elif input.temp_change() == 'substract':
                temp_substr = int(input.temp_substr())
                temp_mult = None
            res = la.run(
                init=init_method, move_choice=str(input.movement()),
                neighbourhood=int(input.neighbourhood()),
                n_shops=int(input.n_shops()), buffer=int(input.buffer()),
                objective=int(input.objective()),
                start_temp=int(input.init_temp()),
                temp_mult=temp_mult,
                temp_substr=temp_substr
            )
            proc = round(la.objective / la.grid['ludnosc'].sum() * 100, 2)
            result.set(f'Population covered: {round(la.objective)} ({proc}%)')
            fig, ax = la.plot_map()
            fig.tight_layout()
    
    @output
    @render.plot(alt="Trend")
    def temp_trend():
        init_temp = input.init_temp()
        values = []
        while init_temp > 100:
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
    def result_objective():
        return str(result())

@module.server
def GA_server(input, output, session):
    initSA = reactive.Value("")
    resultGA = reactive.Value("")

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

            init_method = str(input.init())
            init_method = 'highest' if init_method == 'highest population cells' else 'random'
            if input.temp_change() == 'multiply':
                temp_mult = float(input.temp_mult())
                temp_substr = None
            elif input.temp_change() == 'substract':
                temp_substr = int(input.temp_substr())
                temp_mult = None

            pool = mp.Pool(input.workers())
            params = (
                init_method, str(input.movement()), int(input.neighbourhood()), int(input.n_shops()),
                int(input.buffer()), int(input.objective()), int(input.init_temp()), temp_mult, temp_substr
            )
            result_objects = [pool.apply_async(parallel_annealing, args=(data, i, *params)) for i in range(0, input.iterations())]
            results = [r.get() for r in result_objects]
            pool.close()
            pool.join()

            results.sort(key = lambda x: x[0])
            results = [r for i, r, params in results]
            initSA.set(f'Best score from SA: {round(results[0].objective)}')
            fig, ax = results[0].plot_map()
            fig.tight_layout()

            le = LocationEvolution(data, [res.shops for res in results])
            res = le.run(input.epochs())
            proc = round(max(le.scores) / le.grid['ludnosc'].sum() * 100, 2)
            resultGA.set(f'Population covered: {round(max(le.scores))} ({proc}%)')
            le.plot_map()

    @output
    @render.plot(alt="Trend")
    def temp_trend():
        init_temp = input.init_temp()
        values = []
        while init_temp > 100:
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
        return str(initSA())
    
    @output
    @render.text
    def result_objective_end():
        return str(resultGA())
