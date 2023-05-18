import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import multiprocessing as mp
from shiny import App, render, ui, reactive
from uiModules import SA_ui, GA_ui
from serverModules import SA_server, GA_server
from algorithms import LocationAnnealing, LocationEvolution
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

app_ui = ui.page_fluid(
    ui.h1(ui.HTML('<b>Simmulated annealing of shop locations</b>'), align='center'),
    ui.navset_tab_card(
        SA_ui('SA'),
        GA_ui('GA')
    )
)

def server(input, output, session):
    SA_server('SA')
    GA_server('GA')

app = App(app_ui, server, debug=True)