"""
CNN Pipeline Runner + Presentación Slides
TP Final — Aprendizaje Automático · Radiografías Veterinarias
"""
import os
import streamlit as st
from pathlib import Path
import nbformat
import nbformat.v4 as nbv4
import subprocess
import re
import tempfile
import io

_ANSI = re.compile(r'\x1b\[[0-9;]*[mGKHFABCDJK]')
