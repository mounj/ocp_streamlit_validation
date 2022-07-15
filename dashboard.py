import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import pickle
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
import io
import plotly.express as px
import plotly.graph_objs as go

class State:
    def __init__(self, path='state.pickle', default_state_class=dict):
        self.path = path
        self.default_state_class = default_state_class

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as inf:
                self.state = pickle.load(inf)
        else:
            self.state = self.default_state_class()

    def get_state(self):
        return self.state

    def save(self):
        with open(self.path, 'wb') as outf:
            pickle.dump(self.state, outf)

def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

def app():
    store = State()
    store.load()

    name = store.get_state().get('name', None)
    if name:
        st.text(f'Hello {name}')
    else:
        st.text(f'Please enter your name')
        name_input = st.text_input('your name')
        name = name_input

        if name != '':
            store.get_state()['name'] = name

        store.save()
        next_page = st.button('Next page')
        if next_page:
            rerun()

if __name__ == "__main__":
    app()