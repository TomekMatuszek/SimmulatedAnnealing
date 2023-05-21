from shiny import module, ui
from shiny._namespaces import resolve_id
import multiprocessing as mp

def SAconfiguration():
    ns = resolve_id
    return ui.column(
        3,
        ui.h4('Basic settings', style='text-align: center'),
        ui.input_file(ns("grid_file"), "Upload grid population data", accept=[".gpkg", ".shp"], multiple=False),
        ui.input_selectize(ns("init"), "Initial position of points", {'random_highest':'random highest population cells', 'highest':'highest population cells', 'random':'random cells'}),
        ui.input_selectize(ns("movement"), "Movement scheme", ['random', 'greedy', 'steep']),
        ui.input_numeric(ns("n_shops"), "Number of locations", value=6),
        ui.input_numeric(ns("buffer"), "Radius of influence", value=1500),
        ui.input_slider(ns("neighbourhood"), "Neighbourhood range", 1, 10, 2),
        ui.h5('Termination conditions:'),
        ui.input_checkbox("objective", "Objective value", False),
        ui.input_checkbox("evals", "Number of iterations", False),
        ui.input_checkbox("prop", "Proportion of rejected permutations", False),
        ui.hr(),
        ui.output_ui(ns("term_obj")),
        ui.output_ui(ns("term_eval")),
        ui.output_ui(ns("term_prop")),
        align='center',
        style='background-color: #eeeeee; border-color: #222222'
    ), ui.column(
        3,
        ui.h4('Temperature settings', style='text-align: center'),
        ui.input_numeric(ns("init_temp"), "Initial temperature", value=20000),
        ui.input_selectize(id=ns("temp_change"), label="Temperature change method", choices=['multiply', 'substract']),
        ui.hr(),
        ui.output_ui(ns("mult_substr")),
        ui.hr(),
        ui.output_plot(ns("temp_trend"), height='300px', width='300px'),
        align='center',
        style='background-color: #eeeeee; border-color: #222222'
    )

@module.ui
def SA_ui():
    ns = resolve_id
    return ui.nav(
            "Simple simmulated annealing",
            ui.row(
                ui.column(
                    6, ui.p(ui.input_action_button(ns("run"), "Calculate", class_="btn-primary", width='100%')),
                ),
                ui.column(2),
                ui.column(
                    2, ui.download_button(ns('download_results'), "Download results as CSV")
                ),
                ui.column(2),
            ),
            ui.row(
                SAconfiguration(),
                ui.column(
                    6,
                    ui.navset_tab_card(
                        ui.nav(
                            "Result map",
                            ui.output_plot(ns("map"), height='600px', width='600px'),
                            ui.output_text_verbatim(ns("result_objective")),
                        ),
                        ui.nav(
                            "Plots",
                            ui.output_plot(ns("obj_plot"), height='300px', width='600px'),
                            ui.output_plot(ns("prob_plot"), height='300px', width='600px')
                        )
                    )
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
                        SAconfiguration(),
                        ui.tags.style('#result_objective {text-align: center} #map {margin: auto; display: block;}')
                    )
                ),
                ui.nav(
                    "GA configuration",
                    ui.row(
                        ui.column(
                            3, ui.p(ui.input_action_button(ns("run"), "Calculate", class_="btn-primary", width='100%')),
                        ),
                        ui.column(3),
                        ui.column(3, ui.download_button(ns('download_results'), "Download results as CSV")),
                        ui.column(3)
                    ),
                    ui.row(
                        ui.column(
                            3,
                            ui.h4('SA parallelization settings', style='text-align: center'),
                            ui.input_slider(ns("workers"), "Number of cores to use", 1, mp.cpu_count(), round(mp.cpu_count() / 2)),
                            ui.input_numeric(ns("iterations"), "Number of annealings", value=100),
                            ui.hr(),
                            ui.h4('Genetic algorithm settings', style='text-align: center'),
                            ui.input_numeric(ns("epochs"), "Number of epochs to evaluate", value=1000),
                            align='center',
                            style='background-color: #eeeeee; border-color: #222222'
                        ),
                        ui.column(
                            9,
                            ui.output_plot(ns("map"), height='500px', width='500px'),
                            ui.output_text_verbatim(ns("result_objective_start")),
                            ui.output_text_verbatim(ns("result_objective_end")),
                        ),
                        ui.tags.style('#GA-result_objective_start {text-align: center} #GA-result_objective_end {text-align: center} #GA-map {margin: auto; display: block;}')
                    )
                )
            )
        )