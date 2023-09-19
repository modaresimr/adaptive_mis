import time
from ipywidgets import Button
from jupyter_ui_poll import ui_events
from tqdm.notebook import tqdm


def wait_for_user(ui):
    def_ui = ui.value
    # Wait for user to press the button
    with ui_events() as poll:
        i = 0
        while def_ui == ui.value:
            i += 1
            poll(10)  # React to UI events (upto 10 at a time)
            print('.', end='')
            time.sleep(0.1)
            if (i % 100) == 0:
                print('\r', )
    print('continuing...')
