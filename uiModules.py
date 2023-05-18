from shiny import module, ui
from shiny._namespaces import resolve_id
import multiprocessing as mp

@module.ui
def SA_ui():
    ns = resolve_id
    return ui.nav(
            "Simple simmulated annealing",
            ui.row(
                ui.column(
                    6,
                    ui.p(ui.input_action_button(ns("run"), "Calculate", class_="btn-primary", width='100%')),
                )
            ),
            ui.row(
                ui.column(
                    3,
                    ui.h4('Basic settings', style='text-align: center'),
                    ui.input_file(ns("grid_file"), "Upload grid population data", accept=[".gpkg", ".shp"], multiple=False),
                    ui.input_selectize(ns("init"), "Initial position of points", ['highest population cells', 'random cells']),
                    ui.input_selectize(ns("movement"), "Movement scheme", ['random', 'greedy', 'steep']),
                    ui.input_numeric(ns("n_shops"), "Number of locations", value=6),
                    ui.input_numeric(ns("buffer"), "Radius of influence", value=1500),
                    ui.input_slider(ns("neighbourhood"), "Neighbourhood range", 1, 10, 2),
                    ui.input_numeric(ns("objective"), "Population objective", value=250000),
                    align='center',
                    style='background-color: #eeeeee; border-color: #222222'
                ),
                ui.column(
                    3,
                    ui.h4('Temperature settings', style='text-align: center'),
                    ui.input_numeric(ns("init_temp"), "Initial temperature", value=20000),
                    ui.input_selectize(id=ns("temp_change"), label="Temperature change method", choices=['multiply', 'substract']),
                    ui.output_ui(ns("mult_substr")),
                    ui.output_plot(ns("temp_trend"), height='300px', width='300px'),
                    align='center',
                    style='background-color: #eeeeee; border-color: #222222'
                ),
                ui.column(
                    6,
                    ui.output_plot(ns("map"), height='600px', width='600px'),
                    ui.output_text_verbatim(ns("result_objective")),
                ),
                ui.tags.style('#SA-result_objective {text-align: center} #SA-map {margin: auto; display: block;}')
            )
        )

@module.ui
def GA_ui():
    ns = resolve_id
    return ui.nav("Parallelized SA + Genetic Algorithm",
            ui.navset_tab_card(
                ui.nav(
                    "SA configuration",
                    ui.row(
                        ui.column(
                            3,
                            ui.h4('Basic settings', style='text-align: center'),
                            ui.input_file(ns("grid_file"), "Upload grid population data", accept=[".gpkg", ".shp"], multiple=False),
                            ui.input_selectize(ns("init"), "Initial position of points", ['highest population cells', 'random cells']),
                            ui.input_selectize(ns("movement"), "Movement scheme", ['random', 'greedy', 'steep']),
                            ui.input_numeric(ns("n_shops"), "Number of locations", value=6),
                            ui.input_numeric(ns("buffer"), "Radius of influence", value=1500),
                            ui.input_slider(ns("neighbourhood"), "Neighbourhood range", 1, 10, 2),
                            ui.input_numeric(ns("objective"), "Population objective", value=250000),
                            align='center',
                            style='background-color: #eeeeee; border-color: #222222'
                        ),
                        ui.column(
                            3,
                            ui.h4('Temperature settings', style='text-align: center'),
                            ui.input_numeric(ns("init_temp"), "Initial temperature", value=20000),
                            ui.input_selectize(id=ns("temp_change"), label="Temperature change method", choices=['multiply', 'substract']),
                            ui.output_ui(ns("mult_substr")),
                            ui.output_plot(ns("temp_trend"), height='300px', width='300px'),
                            align='center',
                            style='background-color: #eeeeee; border-color: #222222'
                        ),
                        ui.tags.style('#result_objective {text-align: center} #map {margin: auto; display: block;}')
                    )
                ),
                ui.nav(
                    "GA configuration",
                    ui.row(
                        ui.column(
                            3,
                            ui.p(ui.input_action_button(ns("run"), "Calculate", class_="btn-primary", width='100%')),
                        )
                    ),
                    ui.row(
                        ui.column(
                            3,
                            ui.h4('SA parallelization settings', style='text-align: center'),
                            ui.input_slider(ns("workers"), "Number of cores to use", 1, mp.cpu_count(), 1),
                            ui.input_numeric(ns("iterations"), "Number of iterations", value=100),
                            ui.hr(),
                            ui.h4('Genetic algorithm settings', style='text-align: center'),
                            ui.input_numeric(ns("epochs"), "Number of epochs to evaluate", value=1000),
                            align='center',
                            style='background-color: #eeeeee; border-color: #222222'
                        ),
                        ui.column(
                            9,
                            ui.output_plot(ns("map"), height='600px', width='600px'),
                            ui.output_text_verbatim(ns("result_objective_start")),
                            ui.output_text_verbatim(ns("result_objective_end")),
                        ),
                        ui.tags.style('#GA-result_objective_start {text-align: center} #GA-map {margin: auto; display: block;}')
                    )
                )
            )
        )