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

def main():
    # １
    st.sidebar.title("test_streamlit")
    st.session_state["file"]=st.sidebar.file_uploader("upload", on_change=change_page)

def change_page():
    # 
    st.session_state["page_control"]=1

def next_page():
    # ２
    st.sidebar.title("next_page")
    st.markdown("## next_page")

# 
if ("page_control" in st.session_state and
   st.session_state["page_control"] == 1):
    next_page()
else:
    st.session_state["page_control"] = 0