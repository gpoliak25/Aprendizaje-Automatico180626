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

NOTEBOOK_DIR = Path(__file__).parent
_LOCAL_NBCONV = Path(r"C:\Users\gpoli\venvs\caece-mineria\Scripts\jupyter-nbconvert.exe")
if _LOCAL_NBCONV.exists():
    JUPYTER_NBCONV = _LOCAL_NBCONV
else:
    import shutil as _shutil
    _which = _shutil.which("jupyter-nbconvert") or _shutil.which("jupyter")
    JUPYTER_NBCONV = Path(_which) if _which else Path("jupyter-nbconvert")

CLASS_NAMES = ["ok", "patologica"]
IMG_SIZE    = (224, 224)

PIPELINE = [
    {"pattern": "01_*", "title": "Exploración",      "icon": "🔍", "color": "#7c3aed",
     "desc": "Conteo, balance de clases, resoluciones, imágenes corruptas"},
    {"pattern": "02_*", "title": "Preparación",       "icon": "✂️", "color": "#2563eb",
     "desc": "Split 70/15/15, data augmentation, pesos de clase"},
    {"pattern": "03_*", "title": "CNN desde cero",    "icon": "🧱", "color": "#0891b2",
     "desc": "Arquitectura propia — Conv2D, BatchNorm, Dropout"},
    {"pattern": "04_*", "title": "Transfer Learning", "icon": "🔁", "color": "#059669",
     "desc": "MobileNetV2 con fine-tuning"},
    {"pattern": "05_*", "title": "Evaluación",        "icon": "📊", "color": "#d97706",
     "desc": "Confusion matrix, ROC, F1-score, comparación de modelos"},
    {"pattern": "06_*", "title": "GradCAM",           "icon": "🗺️", "color": "#dc2626",
     "desc": "Mapas de activación para interpretabilidad"},
]
STATUS_ICON  = {"pending": "○", "running": "◉", "success": "●", "error": "✕"}
STATUS_COLOR = {"pending": "#484f58", "running": "#388bfd", "success": "#3fb950", "error": "#f85149"}
STATUS_LABEL = {"pending": "PENDIENTE", "running": "EJECUTANDO", "success": "OK", "error": "ERROR"}

SLIDES_DIR = NOTEBOOK_DIR / "assets" / "slides"
SLIDES = [
    {"title": "Portada",                                        "file": "slide_01.png"},
    {"title": "El Desafío Clínico",                            "file": "slide_02.png"},
    {"title": "El Espacio de Datos",                            "file": "slide_03.png"},
    {"title": "La Métrica Estrella",                            "file": "slide_04.png"},
    {"title": "El Enfrentamiento Analítico",                    "file": "slide_05.png"},
    {"title": "Resultados — La Ilusión de Aprender",           "file": "slide_06.png"},
    {"title": "Resultados — El Poder de la Transferencia",     "file": "slide_07.png"},
    {"title": "Síntesis de Desempeño",                         "file": "slide_08.png"},
    {"title": "Calibrando la Decisión",                        "file": "slide_09.png"},
    {"title": "Abriendo la Caja Negra — Grad-CAM",             "file": "slide_10.png"},
    {"title": "Evidencia Clínica — El Acierto",                "file": "slide_11.png"},
    {"title": "Anatomía del Error",                             "file": "slide_12.png"},
    {"title": "Conclusiones y Evolución",                      "file": "slide_13.png"},
]
N_TOTAL = len(SLIDES)

NAV = [
    ("slides",    "📊", "Presentación"),
    ("pipeline",  "🔄", "Pipeline Runner"),
    ("notebooks", "📄", "Inspeccionar Notebooks"),
    ("pred",      "🩻", "Predicción en Vivo"),
]

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Pipeline · Rx Veterinarias",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#0d1117 !important; }
  .main .block-container { padding-top:1rem; max-width:1340px; }
  *,p,span,label,div { color:#c9d1d9; }
  a { color:#58a6ff !important; }
  h1 { color:#f0f6fc !important; font-size:1.95rem !important; letter-spacing:-.4px; margin:0 !important; }
  h2 { color:#e6edf3 !important; font-size:1.25rem !important; }
  h3 { color:#8b949e !important; }

  [data-testid="stSidebar"] { background:#161b22 !important; border-right:1px solid #21262d; }
  /* Sidebar radio → nav menu */
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    background:#161b22 !important; border:1px solid #21262d !important;
    border-radius:8px !important; padding:8px 12px !important;
    margin-bottom:4px !important; width:100% !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:#21262d !important; border-color:#30363d !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:#1f2d4a !important; border-color:#388bfd !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color:#58a6ff !important; font-weight:600 !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size:.88rem !important; margin:0 !important;
  }

  /* Sidebar reiniciar button */
  [data-testid="stSidebar"] .stButton>button {
    background:#161b22 !important; color:#8b949e !important;
    border:1px solid #21262d !important; border-radius:8px !important;
    font-weight:400 !important; font-size:.82rem !important;
    padding:.45rem 1rem !important; min-height:36px !important;
  }
  [data-testid="stSidebar"] .stButton>button:hover {
    background:#21262d !important; color:#c9d1d9 !important; border-color:#30363d !important; }

  button[data-baseweb="tab"] { background:transparent !important; color:#8b949e !important;
    border-bottom:2px solid transparent !important; font-weight:500; }
  button[data-baseweb="tab"][aria-selected="true"] { color:#e6edf3 !important;
    border-bottom:2px solid #388bfd !important; }
  [data-testid="stTabsContent"] { padding-top:1.2rem; }

  /* Slide image: no white borders, blends into dark background */
  [data-testid="stImage"] { background:#0d1117 !important; }
  [data-testid="stImage"] img { border-radius:10px; display:block; }

  .stButton>button {
    background:#1f6feb !important; color:#fff !important;
    border:1px solid #388bfd !important; border-radius:10px !important;
    font-weight:700 !important; font-size:1rem !important;
    padding:.8rem 1.2rem !important; min-height:52px !important;
    width:100% !important;
  }
  .stButton>button:hover { background:#388bfd !important; }
  .stButton>button:disabled { background:#21262d !important; border-color:#30363d !important;
    color:#484f58 !important; }

  .stTextInput>div>div { background:#161b22 !important; border:1px solid #30363d !important; border-radius:6px; }
  input { color:#e6edf3 !important; }
  .stSelectbox>div>div { background:#161b22 !important; border-color:#30363d !important; }
  [data-testid="stToggle"] { accent-color:#1f6feb; }

  [data-testid="stFileUploader"] {
    background:#161b22 !important; border:2px dashed #30363d !important; border-radius:10px !important; }
  [data-testid="stFileUploaderDropzone"] { background:transparent !important; }

  details { background:#161b22 !important; border:1px solid #21262d !important;
    border-radius:8px !important; margin-bottom:6px !important; }
  summary { color:#e6edf3 !important; font-size:.88rem; padding:8px 12px; }

  pre,code,.stCode { background:#010409 !important; color:#79c0ff !important;
    border:1px solid #21262d !important; border-radius:6px !important; }

  [data-testid="stMetric"] { background:#161b22; border:1px solid #21262d;
    border-radius:8px; padding:.8rem 1rem; }
  [data-testid="stMetricValue"] { color:#58a6ff !important; font-size:1.6rem !important; }
  [data-testid="stMetricLabel"] { color:#8b949e !important; font-size:.78rem !important; }

  .stProgress>div>div { background:#1f6feb !important; border-radius:4px; }

  .stInfo    { background:#0d1b2a !important; border-color:#1f6feb !important; }
  .stSuccess { background:#0d1f12 !important; border-color:#3fb950 !important; }
  .stError   { background:#1f0d0d !important; border-color:#f85149 !important; }
  .stWarning { background:#1f1a0d !important; border-color:#d29922 !important; }

  hr { border-color:#21262d !important; margin:1.2rem 0 !important; }
  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "seccion" not in st.session_state:
    st.session_state.seccion = "pipeline"
if "paso" not in st.session_state:
    st.session_state.paso = 0
if "statuses" not in st.session_state:
    st.session_state.statuses = ["pending"] * len(PIPELINE)
if "logs" not in st.session_state:
    st.session_state.logs = [""] * len(PIPELINE)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def find_nb(pattern: str):
    hits = sorted(NOTEBOOK_DIR.glob(f"{pattern}.ipynb"))
    return hits[0] if hits else None

def cell_count(nb_path) -> int:
    try:
        nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
        return sum(1 for c in nb.cells if c.cell_type == "code" and c.source.strip())
    except Exception:
        return 0

def render_cards(statuses):
    html = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:10px 0 18px;">'
    for i, step in enumerate(PIPELINE):
        s = statuses[i]
        sc, si, sl = STATUS_COLOR[s], STATUS_ICON[s], STATUS_LABEL[s]
        nb = find_nb(step["pattern"])
        cc = cell_count(nb) if nb else "–"
        html += f"""
        <div style="background:#161b22;border:1px solid #21262d;border-top:3px solid {step['color']};
                    border-radius:8px;padding:14px 12px 38px;position:relative;min-height:155px;">
          <div style="font-size:1.55rem;line-height:1">{step['icon']}</div>
          <div style="color:#f0f6fc;font-weight:700;font-size:.88rem;margin:5px 0 3px">{step['title']}</div>
          <div style="color:#8b949e;font-size:.74rem;line-height:1.45">{step['desc']}</div>
          <div style="position:absolute;bottom:10px;left:12px;right:12px;
                      display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#30363d;font-size:.7rem;">{cc} celdas</span>
            <span style="color:{sc};font-size:.72rem;font-weight:700">{si} {sl}</span>
          </div>
        </div>"""
    html += "</div>"
    return html

def render_log(lines):
    visible = lines[-35:]
    body = "<br>".join(
        ln.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").rstrip()
        for ln in visible)
    return ('<div style="background:#010409;border:1px solid #21262d;border-radius:6px;'
            'padding:12px 14px;font-family:Consolas,monospace;font-size:.78rem;'
            'color:#7ee787;max-height:260px;overflow-y:auto;line-height:1.55;">'
            + body + "</div>")

COLAB_STUB = """\
import sys, types as _t
_g=_t.ModuleType("google"); _gc=_t.ModuleType("google.colab")
class _D:
    def mount(self,*a,**k): print("[LOCAL] Drive.mount() omitido")
_gc.drive=_D(); sys.modules["google"]=_g; sys.modules["google.colab"]=_gc
del _g,_gc,_D,_t
"""

def patch_notebook(nb_path, base_dir, mock):
    nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        src = cell.source
        if mock:
            src = src.replace("from google.colab import drive","# [mock] from google.colab import drive")
            src = re.sub(r"drive\.mount\(['\"][^'\"]*['\"]\)","print('[LOCAL] drive.mount omitido')", src)
        if base_dir:
            src = re.sub(r"BASE_DIR\s*=\s*['\"][^'\"]*['\"]",f"BASE_DIR = r'{base_dir}'", src)
        cell.source = src
    if mock:
        nb.cells.insert(0, nbv4.new_code_cell(COLAB_STUB))
    tmp = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w", encoding="utf-8")
    nbformat.write(nb, tmp); tmp.close()
    return tmp.name

def replay_nb_outputs(i, log_ph):
    import time
    step = PIPELINE[i]
    nb_path = find_nb(step["pattern"])
    if nb_path is None:
        return False, f"No encontrado: {step['pattern']}"
    try:
        nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
    except Exception as e:
        return False, str(e)
    log_lines = [f"📄  {nb_path.name}  [MODO DEMO — outputs de Colab]", ""]
    code_cells = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    for j, cell in enumerate(code_cells):
        preview = cell.source.split("\n")[0][:70].strip()
        log_lines.append(f"▶ Celda {j+1}/{len(code_cells)}  — {preview}")
        for out in cell.get("outputs", []):
            otype = out.get("output_type", "")
            if otype == "stream":
                text = "".join(out.get("text", []))
            elif otype in ("execute_result", "display_data"):
                text = "".join(out.get("data", {}).get("text/plain", []))
            elif otype == "error":
                text = f"[ERROR en Colab] {out.get('ename')}: {out.get('evalue')}"
            else:
                text = ""
            for line in text.splitlines()[:15]:
                log_lines.append(f"  {line}")
        log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
        time.sleep(0.05)
    log_lines += ["", f"✓ {nb_path.name} — outputs reproducidos exitosamente."]
    log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
    return True, "\n".join(log_lines)

def execute_nb(i, base_dir, mock, log_ph):
    step = PIPELINE[i]
    nb_path = find_nb(step["pattern"])
    if nb_path is None:
        return False, f"No encontrado: {step['pattern']}"
    tmp_path = patch_notebook(nb_path, base_dir, mock)
    cmd = [str(JUPYTER_NBCONV), "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=600",
           "--ExecutePreprocessor.kernel_name=caece-mineria",
           "--output", str(nb_path), tmp_path]
    log_lines = [f"$ {JUPYTER_NBCONV.name} — {step['title']}", ""]
    log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log_lines.append(_ANSI.sub("", line.rstrip()))
            log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
        proc.wait()
        ok = proc.returncode == 0
    except Exception as e:
        log_lines.append(f"ERROR: {e}"); ok = False
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass
    return ok, "\n".join(log_lines)

@st.cache_resource(show_spinner="Cargando modelo …")
def load_model(path: str):
    import onnxruntime as ort
    return ort.InferenceSession(path)

def predict(model, image_bytes):
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    input_name = model.get_inputs()[0].name
    probs = model.run(None, {input_name: arr})[0][0]
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — navegación vertical siempre visible
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:6px 0 10px">
      <div style="font-size:2rem">🩻</div>
      <div style="color:#e6edf3;font-weight:800;font-size:1rem;margin-top:4px">CNN Pipeline</div>
      <div style="color:#484f58;font-size:.72rem;margin-top:2px">Rx Veterinarias · TP Final</div>
      <div style="color:#484f58;font-size:.68rem;margin-top:1px">L. López · G. Poliak</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Radio como nav — nativamente left-aligned
    nav_labels = [f"{icon}  {label}" for _, icon, label in NAV]
    nav_keys   = [k for k, _, _ in NAV]

    # Inicializar solo si no existe (no pisar el click del usuario en cada rerun)
    if "_nav_radio" not in st.session_state:
        st.session_state["_nav_radio"] = next(
            (f"{icon}  {label}" for k, icon, label in NAV if k == st.session_state.seccion),
            nav_labels[0]
        )

    selected = st.radio("", nav_labels, key="_nav_radio", label_visibility="collapsed")
    new_key = nav_keys[nav_labels.index(selected)]
    if new_key != st.session_state.seccion:
        st.session_state.seccion = new_key
        if new_key == "slides":
            st.session_state.paso = 0
        st.rerun()

    if st.session_state.seccion == "slides":
        st.divider()
        paso_sb = st.session_state.paso
        st.markdown(
            f'<div style="color:#8b949e;font-size:.75rem;margin-bottom:2px">'
            f'Diapositiva {paso_sb+1} / {N_TOTAL}</div>'
            f'<div style="color:#c9d1d9;font-size:.82rem;font-weight:600;margin-bottom:8px">'
            f'{SLIDES[paso_sb]["title"]}</div>',
            unsafe_allow_html=True)
        st.progress((paso_sb + 1) / N_TOTAL)
        if st.button("↩ Reiniciar", key="sidebar_rst", use_container_width=True):
            st.session_state.paso = 0
            st.rerun()

# ─── Header (todas las secciones excepto slides) ───────────────────────────────
seccion = st.session_state.seccion

if seccion != "slides":
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;padding-bottom:14px;">
      <span style="font-size:2.4rem">🩻</span>
      <div>
        <h1>CNN Pipeline · Radiografías Veterinarias</h1>
        <p style="margin:2px 0 0;color:#8b949e;font-size:.9rem;">
          TP Final · Aprendizaje Automático &nbsp;|&nbsp;
          Clasificación <strong style="color:#3fb950">normal</strong> /
          <strong style="color:#f85149">patológico</strong>
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
if seccion == "pipeline":
    cards_ph = st.empty()
    cards_ph.markdown(render_cards(st.session_state.statuses), unsafe_allow_html=True)

    total_cells = sum(cell_count(find_nb(s["pattern"])) for s in PIPELINE if find_nb(s["pattern"]))
    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    mc1.metric("Notebooks",    len(PIPELINE))
    mc2.metric("Total celdas", total_cells)
    mc3.metric("Completados",  sum(1 for s in st.session_state.statuses if s == "success"))
    mc4.metric("Con error",    sum(1 for s in st.session_state.statuses if s == "error"))
    mc5.metric("Pendientes",   sum(1 for s in st.session_state.statuses if s == "pending"))
    st.divider()

    demo_mode = True
    base_dir  = ""
    use_mock  = True
    st.info("**Modo Demo**: reproduce los outputs ya calculados en Colab. "
            "No necesita datos locales ni conexión a Drive.", icon="ℹ️")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    ba, bb, bc = st.columns(3, gap="medium")
    with ba:
        btn_all   = st.button("▶▶  Ejecutar Todo\n\nCorre los 6 notebooks en orden",  use_container_width=True, key="run_all")
    with bb:
        btn_one   = st.button("▶  Ejecutar Uno\n\nCorre el notebook seleccionado",    use_container_width=True, key="run_one")
    with bc:
        btn_reset = st.button("↺  Resetear estados\n\nLimpia todos los indicadores",  use_container_width=True, key="reset")

    nb_sel = st.selectbox("Notebook a ejecutar (botón ▶ Ejecutar Uno):",
        range(len(PIPELINE)), format_func=lambda i: f"{PIPELINE[i]['icon']}  {PIPELINE[i]['title']}",
        key="nb_sel")

    if btn_reset:
        st.session_state.statuses = ["pending"] * len(PIPELINE)
        st.session_state.logs     = [""]        * len(PIPELINE)
        st.rerun()

    run_targets = list(range(len(PIPELINE))) if btn_all else ([nb_sel] if btn_one else None)

    if run_targets is not None:
        prog_ph = st.progress(0.0); status_ph = st.empty(); log_ph = st.empty()
        for seq, idx in enumerate(run_targets):
            step = PIPELINE[idx]
            st.session_state.statuses[idx] = "running"
            cards_ph.markdown(render_cards(st.session_state.statuses), unsafe_allow_html=True)
            status_ph.info(f"⚡  **[{seq+1}/{len(run_targets)}] {step['icon']} {step['title']}** …")
            ok, log = (replay_nb_outputs(idx, log_ph) if demo_mode
                       else execute_nb(idx, base_dir, use_mock, log_ph))
            st.session_state.statuses[idx] = "success" if ok else "error"
            st.session_state.logs[idx]     = log
            prog_ph.progress((seq + 1) / len(run_targets))
            cards_ph.markdown(render_cards(st.session_state.statuses), unsafe_allow_html=True)
        all_ok = all(st.session_state.statuses[i] == "success" for i in run_targets)
        if all_ok:
            status_ph.success("✅ Completado exitosamente.")
        else:
            failed = [PIPELINE[i]["title"] for i in run_targets if st.session_state.statuses[i] == "error"]
            status_ph.error(f"❌ Falló: {', '.join(failed)}")

# ═══════════════════════════════════════════════════════════════════════════════
# INSPECCIONAR NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "notebooks":
    viewer_idx = st.selectbox("Notebook:", range(len(PIPELINE)),
        format_func=lambda i: f"{PIPELINE[i]['icon']}  {PIPELINE[i]['title']}",
        key="viewer_sel")
    nb_path = find_nb(PIPELINE[viewer_idx]["pattern"])
    if nb_path:
        log_content = st.session_state.logs[viewer_idx]
        if log_content.strip():
            with st.expander("📋 Log de la última ejecución"):
                st.code(log_content[-3000:], language=None)
        try:
            nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()
        for j, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                first = cell.source.strip().split("\n")[0]
                if first.startswith("#"):
                    st.markdown(
                        f'<div style="color:#8b949e;font-size:.82rem;margin:10px 0 2px;'
                        f'padding-left:4px;border-left:2px solid #30363d">{first}</div>',
                        unsafe_allow_html=True)
            elif cell.cell_type == "code" and cell.source.strip():
                preview = cell.source.split("\n")[0][:60].strip()
                with st.expander(f"Celda {j+1}  —  `{preview}…`", expanded=(j < 2)):
                    st.code(cell.source, language="python")
                    for out in cell.get("outputs", []):
                        if out.get("output_type") in ("stream", "execute_result"):
                            text = "".join(out.get("text", out.get("data", {}).get("text/plain", [])))
                            if text.strip():
                                st.text(text[:400])
    else:
        st.info("Notebook no encontrado en este directorio.")

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICCIÓN EN VIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "pred":
    st.markdown('<p style="color:#8b949e;margin-bottom:1.2rem;">Cargá un modelo '
                '<code>.onnx</code> y subí una radiografía.</p>', unsafe_allow_html=True)

    default_model_dirs = [NOTEBOOK_DIR/"modelos", NOTEBOOK_DIR/"models", NOTEBOOK_DIR]
    found_models = sorted(set(p for d in default_model_dirs for p in d.glob("*.onnx")))

    col_model, col_btn = st.columns([5, 1])
    with col_model:
        if found_models:
            opts = {p.name: str(p) for p in found_models}
            opts["📂 Otra ruta…"] = "__custom__"
            sel = st.selectbox("Modelo", list(opts.keys()))
            model_path = (st.text_input("Ruta:", placeholder=r"C:\ruta\modelo.onnx")
                          if opts[sel] == "__custom__" else opts[sel])
        else:
            model_path = st.text_input("Ruta al modelo ONNX:", placeholder=r"C:\ruta\modelo.onnx")
    with col_btn:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        st.button("⬇ Cargar", use_container_width=True, key="pred_load")

    model = None
    if model_path and Path(model_path).is_file():
        try:
            model = load_model(model_path)
            st.success(f"✅ Modelo cargado: `{Path(model_path).name}`")
        except Exception as e:
            st.error(f"❌ {e}")
    elif model_path and not Path(model_path).is_file():
        st.warning("⚠️  Archivo no encontrado.")
    if not found_models and not model_path:
        st.info("Copiá la carpeta `modelos/` junto a este archivo, o ingresá la ruta completa al .onnx.")

    st.divider()
    col_upload, col_result = st.columns([1,1], gap="large")
    with col_upload:
        st.markdown("#### Subir radiografía")
        uploaded = st.file_uploader("Imagen", type=["jpg","jpeg","png","bmp","tiff","tif","webp"],
            label_visibility="collapsed", key="pred_uploader")
        if uploaded:
            from PIL import Image as PILImage
            img_bytes = uploaded.read()
            st.image(PILImage.open(io.BytesIO(img_bytes)).convert("RGB"),
                     caption=uploaded.name, use_container_width=True)
            predict_btn = st.button("▶  Predecir", use_container_width=True, key="pred_btn")
        else:
            st.markdown("""
            <div style="background:#161b22;border:2px dashed #30363d;border-radius:10px;
                        padding:40px;text-align:center;color:#484f58;">
              <div style="font-size:2.5rem">🫁</div>
              <div style="margin-top:8px;font-size:.9rem">Arrastrá una Rx aquí</div>
            </div>""", unsafe_allow_html=True)
            predict_btn = False
    with col_result:
        st.markdown("#### Resultado")
        rph = st.empty()
        if uploaded and predict_btn:
            if model is None:
                rph.error("❌ Cargá un modelo primero.")
            else:
                with st.spinner("Analizando …"):
                    try:
                        label, conf, probs = predict(model, img_bytes)
                        is_pat = label == "patologica"
                        clr = "#f85149" if is_pat else "#3fb950"
                        em  = "⚠️" if is_pat else "✅"
                        tit = "PATOLÓGICA" if is_pat else "NORMAL (OK)"
                        p_ok=probs[0]*100; p_pat=probs[1]*100
                        rph.markdown(f"""
                        <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                                    padding:28px 24px;border-top:4px solid {clr};">
                          <div style="font-size:2.8rem;margin-bottom:8px">{em}</div>
                          <div style="color:{clr};font-size:2rem;font-weight:800;margin-bottom:4px">{tit}</div>
                          <div style="color:#8b949e;font-size:.85rem;margin-bottom:20px">
                            confianza: <strong style="color:{clr}">{conf*100:.1f}%</strong></div>
                          <div style="margin-bottom:10px">
                            <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:4px">
                              <span>Normal</span><span style="color:#3fb950">{p_ok:.1f}%</span></div>
                            <div style="background:#21262d;border-radius:4px;height:8px">
                              <div style="background:#3fb950;border-radius:4px;height:8px;width:{p_ok:.1f}%"></div></div>
                          </div>
                          <div>
                            <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:4px">
                              <span>Patológica</span><span style="color:#f85149">{p_pat:.1f}%</span></div>
                            <div style="background:#21262d;border-radius:4px;height:8px">
                              <div style="background:#f85149;border-radius:4px;height:8px;width:{p_pat:.1f}%"></div></div>
                          </div>
                          <div style="margin-top:20px;padding-top:16px;border-top:1px solid #21262d;
                                      color:#484f58;font-size:.75rem;">Modelo: {Path(model_path).name}</div>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        rph.error(f"❌ {e}")
        else:
            rph.markdown("""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                        padding:40px;text-align:center;color:#484f58;">El resultado aparecerá aquí</div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PRESENTACIÓN HTML
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "pres_html":
    import plotly.graph_objects as go

    def card(content, border_color="#1f6feb", pad="22px 24px"):
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #21262d;'
            f'border-left:4px solid {border_color};border-radius:10px;'
            f'padding:{pad};margin-bottom:14px">{content}</div>',
            unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1117 60%,#0d1f2d);
                border:1px solid #21262d;border-radius:14px;padding:36px 40px;
                margin-bottom:24px;text-align:center;">
      <div style="color:#8b949e;font-size:.82rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
        Trabajo Práctico Integrador · Maestría en Ciencia de Datos e Innovación Empresarial</div>
      <h2 style="color:#e6edf3;font-size:1.7rem;font-weight:800;line-height:1.3;margin:10px 0 6px;">
        Detección de Patologías en Radiografías de Tórax Veterinarias<br>
        <span style="color:#388bfd">mediante Redes Neuronales Convolucionales</span></h2>
      <div style="color:#8b949e;font-size:.9rem;margin:10px 0 18px">
        Un análisis del costo del error y la interpretabilidad clínica</div>
      <div style="color:#c9d1d9;font-size:.88rem">
        Lic. Lorena López &nbsp;·&nbsp; Lic. Gisela Poliak &nbsp;·&nbsp;
        Universidad CAECE &nbsp;·&nbsp; Junio 2026</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("## El Desafío Clínico")
    card("<span style='color:#e6edf3'>La interpretación de radiografías es "
         "<strong>vulnerable a la fatiga</strong> y la carga de trabajo. "
         "La IA actúa como filtro de triage.</span>", border_color="#388bfd")
    st.divider()

    st.markdown("## El Espacio de Datos")
    col_pie, col_split = st.columns([1,1])
    with col_pie:
        fig_pie = go.Figure(go.Pie(labels=["Normal (ok)","Patológica"],values=[245,196],
            hole=0.55,marker_colors=["#58a6ff","#d29922"],textinfo="label+percent",textfont_size=13))
        fig_pie.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font_color="#c9d1d9",
            margin=dict(t=20,b=20,l=0,r=0),showlegend=False,height=280,
            annotations=[dict(text="<b>441</b><br>imágenes",x=0.5,y=0.5,
                              font_size=16,font_color="#e6edf3",showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_html")
    with col_split:
        for split,n,pct,color in [("Train",308,"70%","#388bfd"),("Val",66,"15%","#d29922"),("Test",67,"15%","#3fb950")]:
            st.markdown(f'<div style="background:#161b22;border:1px solid #21262d;border-left:4px solid {color};'
                        f'border-radius:8px;padding:12px 16px;margin-bottom:8px;'
                        f'display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="color:{color};font-weight:700">{split}</span>'
                        f'<span style="color:#e6edf3;font-size:1.1rem;font-weight:800">{n}</span>'
                        f'<span style="color:#8b949e">{pct}</span></div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("## La Métrica Estrella: Costo Asimétrico del Error")
    card("<strong style='color:#e6edf3'>Accuracy es engañosa.</strong> "
         "<span style='color:#c9d1d9'> Maximizamos <strong>Recall</strong>: preferimos "
         "falsas alarmas antes que ignorar pacientes enfermos.</span>", border_color="#f85149")
    st.divider()

    st.markdown("## Resultados: CNN vs Transfer Learning")
    col_r1, col_r2 = st.columns(2, gap="large")
    def confusion_fig(tn,fp,fn,tp,title,color):
        fig = go.Figure(go.Heatmap(z=[[tn,fp],[fn,tp]],text=[[str(tn),str(fp)],[str(fn),str(tp)]],
            texttemplate="%{text}",textfont={"size":22,"color":"white"},
            colorscale=[[0,"#0d1117"],[1,color]],showscale=False,
            x=["ok","patologica"],y=["ok","patologica"]))
        fig.update_layout(title=dict(text=title,font_color="#c9d1d9",font_size=13,x=0.5),
            paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font_color="#c9d1d9",
            height=220,margin=dict(t=36,b=30,l=60,r=10),
            xaxis=dict(title="Predicción",color="#8b949e",gridcolor="#21262d"),
            yaxis=dict(title="Real",color="#8b949e",gridcolor="#21262d"))
        return fig
    with col_r1:
        st.plotly_chart(confusion_fig(0,37,0,30,"CNN desde cero — AUC 0.451","#1f6feb"),
                        use_container_width=True, key="cm_cnn_html")
        card("<span style='color:#f85149'>Todo clasificado como patológico. Clínicamente inútil.</span>",
             border_color="#f85149", pad="12px 14px")
    with col_r2:
        st.plotly_chart(confusion_fig(31,6,7,23,"Transfer Learning — AUC 0.852","#3fb950"),
                        use_container_width=True, key="cm_tl_html")
        card("<span style='color:#3fb950'>AUC = 0.852. Discriminación real.</span>",
             border_color="#3fb950", pad="12px 14px")
    st.divider()

    st.markdown("## Síntesis de Desempeño")
    col_tab, col_roc = st.columns([1,1], gap="large")
    with col_tab:
        st.markdown("""<table style="width:100%;border-collapse:collapse;font-size:.84rem">
          <thead><tr style="color:#8b949e;border-bottom:1px solid #21262d">
            <th style="padding:8px;text-align:left">Modelo</th>
            <th style="padding:8px;text-align:center">Recall</th>
            <th style="padding:8px;text-align:center">Prec</th>
            <th style="padding:8px;text-align:center">AUC</th></tr></thead>
          <tbody>
            <tr style="color:#484f58;border-bottom:1px solid #21262d">
              <td style="padding:8px">CNN desde cero</td>
              <td style="padding:8px;text-align:center">1.000*</td>
              <td style="padding:8px;text-align:center">0.448</td>
              <td style="padding:8px;text-align:center">0.451</td></tr>
            <tr style="background:#0d2d1a;color:#3fb950;font-weight:700">
              <td style="padding:10px 8px">Transfer Learning ✓</td>
              <td style="padding:10px 8px;text-align:center">0.867</td>
              <td style="padding:10px 8px;text-align:center">0.793</td>
              <td style="padding:10px 8px;text-align:center">0.852</td></tr>
          </tbody></table>
          <div style="color:#484f58;font-size:.72rem;margin-top:4px">* clasifica todo como positivo</div>
        """, unsafe_allow_html=True)
    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,0.5,0.6,0.8,1],y=[0,0.45,0.5,0.55,1],
            mode="lines",name="CNN (0.451)",line=dict(color="#388bfd",width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,0.05,0.1,0.2,0.4,1],y=[0,0.6,0.77,0.87,0.97,1],
            mode="lines",name="Transfer (0.852)",line=dict(color="#d29922",width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            line=dict(color="#484f58",dash="dash",width=1),showlegend=False))
        fig_roc.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font_color="#c9d1d9",
            height=220,margin=dict(t=10,b=40,l=40,r=10),
            xaxis=dict(title="FPR",gridcolor="#21262d",zeroline=False),
            yaxis=dict(title="Recall",gridcolor="#21262d",zeroline=False),
            legend=dict(bgcolor="#0d1117",bordercolor="#21262d",borderwidth=1,font_size=11,x=0.4,y=0.15))
        st.plotly_chart(fig_roc, use_container_width=True, key="roc_html")
    st.divider()

    st.markdown("## Calibrando la Decisión: Umbral 0.30")
    col_u1, col_u2 = st.columns(2, gap="large")
    for col,umbral,recall,prec,highlight,color in [
        (col_u1,"0.50","0.767","0.793",False,"#388bfd"),
        (col_u2,"0.30","0.867","0.650",True, "#d29922")]:
        with col:
            brd = f"border:2px solid {color}" if highlight else "border:1px solid #21262d"
            st.markdown(f"""<div style="background:#161b22;{brd};border-radius:10px;padding:20px;text-align:center">
              <div style="color:{color};font-size:1.5rem;font-weight:800">Umbral {umbral}</div>
              <div style="display:flex;justify-content:space-around;margin-top:14px">
                <div><div style="color:#d29922;font-size:1.4rem;font-weight:800">{recall}</div>
                     <div style="color:#8b949e;font-size:.78rem">Recall</div></div>
                <div><div style="color:#388bfd;font-size:1.4rem;font-weight:800">{prec}</div>
                     <div style="color:#8b949e;font-size:.78rem">Precision</div></div>
              </div>
              {'<div style="margin-top:12px;background:#2d1e00;border:1px solid #d29922;border-radius:6px;padding:8px;color:#d29922;font-size:.78rem">⭐ Seleccionado</div>' if highlight else ''}
            </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("## Grad-CAM y Análisis de Errores")
    card("Grad-CAM visualiza qué píxeles influyeron más en la decisión. "
         "Validamos que la red observa estructuras anatómicas plausibles, no atajos espurios.",
         border_color="#388bfd")
    st.divider()

    st.markdown("## Conclusiones y Evolución")
    card("🏆 <strong style='color:#3fb950'>Transfer Learning (MobileNetV2, umbral 0.30)</strong> "
         "<span style='color:#c9d1d9'>construye un filtro de triage interpretable que maximiza "
         "el Recall con un dataset médico pequeño.</span>", border_color="#3fb950")

    _, cbtn, _ = st.columns([3,2,3])
    with cbtn:
        if st.button("🎤 Ver Slides PNG", use_container_width=True, key="go_slides_from_pres"):
            st.session_state.seccion = "slides"
            st.session_state.paso = 0
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDES PNG — sin selectbox, sin conflictos de estado
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "slides":
    import plotly.graph_objects as go

    paso = st.session_state.paso

    st.markdown(
        f'<div style="color:#484f58;font-size:.72rem;margin-bottom:6px;">'
        f'🩻 Slides &nbsp;·&nbsp; <span style="color:#8b949e">{SLIDES[paso]["title"]}</span></div>',
        unsafe_allow_html=True)

    slide = SLIDES[paso]
    img_path = SLIDES_DIR / slide["file"]
    if img_path.exists():
        import base64
        with open(img_path, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            f'<div style="background:#0d1117;border-radius:12px;overflow:hidden;'
            f'box-shadow:0 0 40px rgba(0,0,0,0.6);margin:0 auto;">'
            f'<img src="data:image/png;base64,{_b64}" '
            f'style="width:100%;display:block;border-radius:10px;" />'
            f'</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#161b22;border:2px dashed #30363d;border-radius:14px;
                    padding:80px 40px;text-align:center;min-height:380px;">
          <div style="color:#484f58;font-size:.85rem;margin-bottom:12px">Diapositiva {paso+1} / {N_TOTAL}</div>
          <div style="color:#e6edf3;font-size:2rem;font-weight:800;margin-bottom:16px">{slide['title']}</div>
          <div style="color:#484f58;font-size:.8rem">
            Colocá <code style="color:#58a6ff">assets/slides/{slide['file']}</code>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Navegación sin selectbox = sin conflictos de estado ───────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.progress((paso + 1) / N_TOTAL)

    cp, ci, cn = st.columns([1, 5, 1])
    with cp:
        if st.button("◀ Anterior", disabled=(paso == 0), use_container_width=True, key="slide_prev"):
            st.session_state.paso -= 1
            st.rerun()
    with ci:
        st.markdown(
            f'<div style="text-align:center;color:#8b949e;font-size:.82rem;padding:.4rem 0;">'
            f'Diapositiva <strong style="color:#e6edf3">{paso+1}</strong> / {N_TOTAL}'
            f' &nbsp;·&nbsp; <em>{SLIDES[paso]["title"]}</em></div>',
            unsafe_allow_html=True)
    with cn:
        if paso < N_TOTAL - 1:
            if st.button("Siguiente ▶", use_container_width=True, key="slide_next"):
                st.session_state.paso += 1
                st.rerun()
        else:
            if st.button("↩ Inicio", use_container_width=True, key="slide_restart"):
                st.session_state.paso = 0
                st.rerun()
