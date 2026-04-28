"""Workforce Intelligence Dashboard — Streamlit app.

Run with:
    cd project
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from infer import (load_artifacts, predict_skills, recommend_roles,
                   DIM_COLORS, DIM_LABELS)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Workforce Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 8px;
}
.score-bar-bg {
    background: #2e2e3e; border-radius: 4px; height: 8px; margin-top: 4px;
}
.skill-tag {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts (cached across reruns) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and graph...")
def get_artifacts():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_artifacts(device=device)


# ── PyVis graph renderer ─────────────────────────────────────────────────────
def render_graph(occ_label: str, skills: list[dict], height: int = 480) -> str:
    from pyvis.network import Network

    net = Network(height=f"{height}px", width="100%",
                  bgcolor="#0e1117", font_color="#ffffff",
                  directed=False)
    net.set_options("""
    {
      "physics": {"barnesHut": {"gravitationalConstant": -8000, "springLength": 160}},
      "edges": {"smooth": {"type": "continuous"}},
      "interaction": {"hover": true}
    }
    """)

    # Center occupation node
    net.add_node(occ_label, label=occ_label, size=36,
                 color="#e94560", font={"size": 14, "bold": True},
                 title=occ_label, shape="dot")

    max_score = max(s["score"] for s in skills) if skills else 1.0
    for s in skills:
        norm = s["score"] / max_score
        size = 10 + 18 * norm
        net.add_node(
            s["name"],
            label=s["display_name"],
            size=size,
            color=s["color"],
            title=f"{s['dim_label']}: {s['display_name']}\nScore: {s['score']:.3f}",
            shape="dot",
        )
        net.add_edge(occ_label, s["name"],
                     width=1 + 3 * norm,
                     color={"color": s["color"], "opacity": 0.7})

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w",
                                     encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)
    return html


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Workforce Intelligence")
    st.caption("GraphSAGE-powered skill & career recommendation")
    st.divider()
    mode = st.radio("Mode", ["Skill Explorer", "Career Advisor"],
                    label_visibility="collapsed")
    st.divider()

    # Legend
    st.markdown("**Feature Dimensions**")
    for dim, color in DIM_COLORS.items():
        label = DIM_LABELS.get(dim, dim)
        st.markdown(
            f'<span class="skill-tag" style="background:{color};color:#111">'
            f'● {label}</span>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Data: O*NET Release 30.1 · 894 occupations · 160 features")


# ── Mode 1: Skill Explorer ────────────────────────────────────────────────────
if mode == "Skill Explorer":
    st.header("What skills does a role require?")
    st.caption("Search an existing occupation or add a new one — the GNN handles both.")

    arts = get_artifacts()
    occ_title_map = arts["maps"].get("occ_title", {})
    all_titles = sorted(set(occ_title_map.values()))

    col_input, col_k = st.columns([4, 1])
    with col_input:
        selected_title = st.selectbox(
            "Search existing occupation",
            options=["— Add a new occupation below —"] + all_titles,
            index=0,
        )
    with col_k:
        top_k = st.slider("Top K", 5, 25, 12)

    is_new = selected_title == "— Add a new occupation below —"
    new_title, description = "", ""
    if is_new:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            new_title = st.text_input("New occupation title",
                                      placeholder="e.g. AI Policy Analyst")
        with col_b:
            description = st.text_area(
                "Brief description (optional but improves accuracy)",
                placeholder="e.g. Develops AI governance frameworks, advises on regulation...",
                height=80,
            )

    run = st.button("Predict Skills", type="primary", use_container_width=False)

    if run:
        occ_label = new_title.strip() if is_new else selected_title
        if not occ_label:
            st.warning("Please enter an occupation title.")
            st.stop()

        # Resolve existing occupation index
        existing_idx = None
        if not is_new:
            code2idx = arts["maps"]["occ2idx"]
            title2code = {v: k for k, v in occ_title_map.items()}
            code = title2code.get(selected_title)
            if code:
                existing_idx = code2idx.get(code)

        with st.spinner("Running GraphSAGE inference..."):
            skills, is_cold = predict_skills(
                occ_label, description, top_k, arts,
                existing_occ_idx=existing_idx,
            )

        if not skills:
            st.error("No predictions returned. Try a different description.")
            st.stop()

        # Banner
        if is_cold:
            st.success(
                f"**Cold-start prediction** — '{occ_label}' was not in training data. "
                "GraphSAGE used TF-IDF text features to infer skills inductively."
            )
        else:
            st.info(f"Showing top {top_k} predicted skills for **{occ_label}**")

        graph_col, list_col = st.columns([3, 2])

        with graph_col:
            st.subheader("Knowledge Graph")
            html = render_graph(occ_label, skills)
            st.components.v1.html(html, height=500, scrolling=False)

        with list_col:
            st.subheader("Ranked Skills")
            max_score = max(s["score"] for s in skills)
            for i, s in enumerate(skills, 1):
                pct = int(100 * s["score"] / max_score)
                st.markdown(
                    f'<div class="metric-card">'
                    f'<b>{i}. {s["display_name"]}</b> '
                    f'<span class="skill-tag" style="background:{s["color"]};color:#111;float:right">'
                    f'{s["dim_label"]}</span>'
                    f'<div class="score-bar-bg">'
                    f'<div style="width:{pct}%;background:{s["color"]};'
                    f'height:8px;border-radius:4px"></div></div>'
                    f'<small style="color:#888">score {s["score"]:.3f}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Baseline comparison for cold-start
            if is_cold:
                st.divider()
                st.subheader("Why baselines fail here")
                for b in ["Cosine (CF)", "Jaccard (CF)", "Node2Vec"]:
                    st.error(f"**{b}** — No prediction possible (unseen occupation)")
                st.success("**GraphSAGE** — Predicted above ✓ (inductive generalization)")


# ── Mode 2: Career Advisor ────────────────────────────────────────────────────
else:
    st.header("What roles match your skillset?")
    st.caption("Select skills you have — we'll find the best-matching occupations.")

    arts = get_artifacts()
    skill_names = arts["skill_names"]

    # Group skills by dimension for organised multiselect
    from infer import _skill_dimension, _skill_display
    dims = {}
    for s in skill_names:
        d = _skill_dimension(s)
        dims.setdefault(d, []).append(s)

    top_k_roles = st.slider("Number of roles to show", 3, 15, 6)

    st.markdown("**Select your skills** (grouped by dimension)")
    selected = []
    cols = st.columns(len(dims))
    for col, (dim, skills_in_dim) in zip(cols, dims.items()):
        with col:
            color = DIM_COLORS.get(dim, "#aaa")
            label = DIM_LABELS.get(dim, dim)
            st.markdown(
                f'<span class="skill-tag" style="background:{color};color:#111">'
                f'● {label}</span>',
                unsafe_allow_html=True,
            )
            chosen = st.multiselect(
                label,
                options=skills_in_dim,
                format_func=_skill_display,
                label_visibility="collapsed",
                key=f"ms_{dim}",
            )
            selected.extend(chosen)

    st.divider()

    if selected:
        st.markdown(f"**{len(selected)} skill(s) selected** — finding best roles...")

        with st.spinner("Computing matches..."):
            roles = recommend_roles(selected, top_k_roles, arts)

        if not roles:
            st.warning("No roles found. Try selecting more skills.")
        else:
            max_score = max(r["score"] for r in roles)
            for i, r in enumerate(roles, 1):
                pct = int(100 * r["score"] / max_score)
                with st.container():
                    r_col1, r_col2 = st.columns([3, 1])
                    with r_col1:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<b style="font-size:16px">#{i} {r["title"]}</b><br>'
                            f'<small style="color:#888">{r["code"]}</small>'
                            f'<div class="score-bar-bg">'
                            f'<div style="width:{pct}%;background:#e94560;'
                            f'height:8px;border-radius:4px"></div></div>'
                            f'<small style="color:#888">match score {r["score"]:.3f}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with r_col2:
                        if st.button(f"Explore skills →", key=f"explore_{i}"):
                            st.session_state["explore_occ_idx"] = r["occ_idx"]
                            st.session_state["explore_occ_title"] = r["title"]

            # If user clicked Explore, show skill graph inline
            if "explore_occ_idx" in st.session_state:
                idx = st.session_state["explore_occ_idx"]
                title = st.session_state["explore_occ_title"]
                st.divider()
                st.subheader(f"Skills for: {title}")
                skills, _ = predict_skills(title, "", 15, arts, existing_occ_idx=idx)
                g_col, l_col = st.columns([3, 2])
                with g_col:
                    html = render_graph(title, skills, height=420)
                    st.components.v1.html(html, height=440, scrolling=False)
                with l_col:
                    for s in skills[:10]:
                        st.markdown(
                            f'<span class="skill-tag" style="background:{s["color"]};'
                            f'color:#111">● {s["display_name"]}</span>',
                            unsafe_allow_html=True,
                        )
    else:
        st.info("Select at least one skill above to get role recommendations.")
