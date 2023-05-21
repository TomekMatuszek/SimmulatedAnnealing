from shiny import App, ui
from uiModules import SA_ui, GA_ui
from serverModules import SA_server, GA_server
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

app_ui = ui.page_fluid(
    ui.h1(ui.HTML('<b>Simmulated annealing of shop locations</b>'), align='center'),
    ui.navset_tab_card(
        SA_ui('SA'),
        GA_ui('GA'),
        ui.nav_spacer(),
        ui.nav_control(
            ui.a(
                "GitHub repo",
                href="https://github.com/TomekMatuszek/SimmulatedAnnealing",
                target="_blank",
            )
        )
    )
)

def server(input, output, session):
    SA_server('SA')
    GA_server('GA')

app = App(app_ui, server, debug=True)