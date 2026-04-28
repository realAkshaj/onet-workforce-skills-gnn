"""Workforce Intelligence Dashboard.

Run:
    cd project
    C:\\Python313\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

SESSION_FILE = ROOT / "demo_session.json"


def _save_session():
    data = {str(k): v for k, v in
            st.session_state.get("graph_updates", {}).items()}
    SESSION_FILE.write_text(json.dumps(data, indent=2))


def _load_session():
    if SESSION_FILE.exists():
        raw = json.loads(SESSION_FILE.read_text())
        st.session_state["graph_updates"] = {int(k): v for k, v in raw.items()}
        return True
    return False


# Auto-load saved session once on startup
if "session_loaded" not in st.session_state:
    _load_session()
    st.session_state["session_loaded"] = True

from infer import (load_artifacts, predict_skills, predict_skills_with_update,
                   recommend_roles, find_similar_roles,
                   DIM_COLORS, DIM_LABELS, _skill_display, _skill_dimension)

st.set_page_config(
    page_title="Workforce Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS — only for the score bar and skill tags
st.markdown("""
<style>
.bar-bg { background:#2e2e3e; border-radius:4px; height:8px; margin:4px 0 8px 0; }
.bar-fill { height:8px; border-radius:4px; }
.dim-tag {
    display:inline-block; padding:2px 10px; border-radius:10px;
    font-size:12px; font-weight:600; margin:2px 2px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_artifacts(device=device)


def draw_graph(occ_label: str, skills: list[dict], height: int = 460,
               injected: set[str] | None = None):
    """Return a Plotly figure — radial spoke layout, occupation at centre."""
    import plotly.graph_objects as go
    import math

    n = len(skills)
    if n == 0:
        return go.Figure()

    mx = max(s["score"] for s in skills)

    # Node coordinates: occupation at origin, skills on a circle
    node_x, node_y, node_color, node_size = [0.0], [0.0], ["#e94560"], [28]
    node_hover = [f"<b>{occ_label}</b>"]
    node_label = [occ_label]

    edge_x, edge_y, edge_color = [], [], []

    for i, s in enumerate(skills):
        angle = 2 * math.pi * i / n
        r = 1.0
        x, y = r * math.cos(angle), r * math.sin(angle)
        is_inj = bool(injected and s["name"] in injected)
        color  = "#f9c74f" if is_inj else s["color"]
        norm   = s["score"] / mx if mx > 0 else 0.5

        node_x.append(x);  node_y.append(y)
        node_color.append(color)
        node_size.append(int(10 + 16 * norm))
        prefix = "+ " if is_inj else ""
        node_label.append(prefix + s["display_name"])
        node_hover.append(
            f"<b>{'[ADDED] ' if is_inj else ''}{s['display_name']}</b>"
            f"<br>{s['dim_label']}<br>score {s['score']:.3f}"
        )
        # Edge from centre to skill
        edge_x += [0, x, None];  edge_y += [0, y, None]
        edge_color.append("rgba(249,199,79,0.8)" if is_inj else
                          s["color"].replace("#", "rgba(") + ",0.5)")

    fig = go.Figure()

    # Draw all edges first
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#444"),
        hoverinfo="none", showlegend=False,
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=1.5, color="#222")),
        text=node_label,
        textposition="top center",
        textfont=dict(color="#e0e0e0", size=10),
        hovertext=node_hover, hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x"),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Workforce Intelligence")
    st.caption("O*NET graph-based skill recommendation")
    st.divider()
    mode = st.radio("Select mode", ["Skill Explorer", "Career Advisor"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**Dimension legend**")
    for dim, color in DIM_COLORS.items():
        st.markdown(
            f'<span class="dim-tag" style="background:{color};color:#111">'
            f'{DIM_LABELS[dim]}</span>',
            unsafe_allow_html=True,
        )
    st.divider()

    # Session persistence
    st.markdown("**Demo session**")
    updates = st.session_state.get("graph_updates", {})
    if updates:
        arts_s = get_artifacts()
        occ_tm = arts_s["maps"].get("occ_title", {})
        for oi, snames in updates.items():
            code = arts_s["maps"]["idx2occ"][oi]
            title = occ_tm.get(code, code)
            st.caption(f"+ {_skill_display(snames[0])} → {title}" if snames else title)

    sc1, sc2 = st.columns(2)
    if sc1.button("Save", use_container_width=True):
        _save_session()
        st.toast("Session saved.")
    if sc2.button("Clear", use_container_width=True):
        st.session_state.pop("graph_updates", None)
        st.session_state.pop("se", None)
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
        st.toast("Session cleared.")
        st.rerun()

    if SESSION_FILE.exists():
        st.caption(f"Saved: {SESSION_FILE.name}")

    st.divider()
    st.caption("O*NET Release 30.1 · 894 occupations · 160 features")


# ── Mode 1: Skill Explorer ────────────────────────────────────────────────────
if mode == "Skill Explorer":
    import time, datetime, os as _os

    st.header("Skill Explorer")
    st.write("Select an existing occupation or describe a new one to see predicted required skills.")

    arts = get_artifacts()
    occ_title_map = arts["maps"].get("occ_title", {})
    all_titles = sorted(set(occ_title_map.values()))

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_title = st.selectbox(
            "Search occupation",
            options=["-- Add a new occupation --"] + all_titles,
        )
    with col2:
        top_k = st.slider("Top K skills", 5, 20, 10)

    is_new = selected_title == "-- Add a new occupation --"
    new_title, description = "", ""
    if is_new:
        new_title = st.text_input("Occupation title", placeholder="e.g. AI Policy Analyst")
        description = st.text_area(
            "Brief description (optional — improves accuracy for new roles)",
            placeholder="e.g. Develops AI governance frameworks, advises on regulation...",
            height=80,
        )

    if st.button("Predict skills", type="primary"):
        occ_label = new_title.strip() if is_new else selected_title
        if not occ_label:
            st.warning("Please enter an occupation title.")
            st.stop()

        existing_idx = None
        if not is_new:
            title2code = {v: k for k, v in occ_title_map.items()}
            code = title2code.get(selected_title)
            if code:
                existing_idx = arts["maps"]["occ2idx"].get(code)

        t0 = time.perf_counter()
        with st.spinner("Running inference..."):
            skills, is_cold, cold_text = predict_skills(
                occ_label, description, top_k, arts,
                existing_occ_idx=existing_idx,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not skills:
            st.error("No predictions returned.")
            st.stop()

        # Store in session state so UI persists across reruns
        st.session_state["se"] = {
            "occ_label": occ_label, "skills": skills,
            "is_cold": is_cold, "cold_text": cold_text,
            "existing_idx": existing_idx, "top_k": top_k,
            "elapsed_ms": elapsed_ms,
        }

    # ── Render from session state (persists after multiselect reruns) ─────────
    if "se" in st.session_state:
        p = st.session_state["se"]
        occ_label   = p["occ_label"]
        skills      = p["skills"]
        is_cold     = p["is_cold"]
        cold_text   = p["cold_text"]
        existing_idx = p["existing_idx"]
        top_k       = p["top_k"]
        elapsed_ms  = p["elapsed_ms"]

        if is_cold:
            ckpt = ROOT / "checkpoints" / "sage_coldstart.pt"
            trained_on = datetime.datetime.fromtimestamp(
                _os.path.getmtime(ckpt)).strftime("%b %d %Y") if ckpt.exists() else "N/A"
            m1, m2, m3 = st.columns(3)
            m1.metric("Inference time", f"{elapsed_ms:.0f} ms")
            m2.metric("Model retrained?", "No")
            m3.metric("Checkpoint trained", trained_on)
            st.info(
                f"'{occ_label}' was not in training data. "
                "GraphSAGE scores it in a single forward pass — no retraining required."
            )

        graph_col, list_col = st.columns([3, 2])
        with graph_col:
            st.subheader("Knowledge graph")
            st.plotly_chart(draw_graph(occ_label, skills), width="stretch")

        with list_col:
            st.subheader("Predicted skills")
            mx = max(s["score"] for s in skills)
            for i, s in enumerate(skills, 1):
                pct = int(100 * s["score"] / mx) if mx > 0 else 50
                ca, cb = st.columns([5, 2])
                with ca:
                    st.markdown(f"**{i}. {s['display_name']}**")
                with cb:
                    st.markdown(
                        f'<span class="dim-tag" style="background:{s["color"]};color:#111">'
                        f'{s["dim_label"]}</span>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;'
                    f'background:{s["color"]}"></div></div>', unsafe_allow_html=True)

        # ── Add a skill (existing occupations only) ───────────────────────────
        if not is_cold and existing_idx is not None:
            st.divider()
            st.subheader("Add a skill to this role")
            st.caption(
                "Inject a new skill into the graph and re-run inference in one "
                "forward pass. Model parameters are not changed."
            )

            known = set(int(i) for i in arts["train_mask"][existing_idx].nonzero()[0])
            available = [arts["skill_names"][i] for i in range(arts["n_skill"])
                         if i not in known]

            added = st.multiselect(
                "Select skills to add",
                options=available,
                format_func=_skill_display,
                key=f"add_ms_{existing_idx}",
            )

            # Always sync graph_updates so Career Advisor sees the change
            if "graph_updates" not in st.session_state:
                st.session_state["graph_updates"] = {}
            if added:
                st.session_state["graph_updates"][existing_idx] = added
            else:
                st.session_state["graph_updates"].pop(existing_idx, None)

            if added:
                with st.spinner("Updating graph..."):
                    updated_skills, update_ms = predict_skills_with_update(
                        existing_idx, added, top_k, arts)

                u1, u2, u3 = st.columns(3)
                u1.metric("Update time", f"{update_ms:.0f} ms")
                u2.metric("Skills added", len(added))
                u3.metric("Model retrained?", "No")

                orig_names = {s["name"] for s in skills}
                upd_names  = {s["name"] for s in updated_skills}

                g_col, diff_col = st.columns([3, 2])
                with g_col:
                    st.subheader("Updated knowledge graph")
                    injected_set = set(added)
                    graph_skills = list(updated_skills) + [
                        {"name": sn, "display_name": _skill_display(sn),
                         "dimension": _skill_dimension(sn),
                         "dim_label": DIM_LABELS.get(_skill_dimension(sn), ""),
                         "color": "#f9c74f", "score": 0.85}
                        for sn in added
                    ]
                    st.plotly_chart(
                        draw_graph(occ_label, graph_skills, injected=injected_set),
                        width="stretch")

                with diff_col:
                    st.subheader("What changed")
                    new_in_top = upd_names - orig_names
                    dropped    = orig_names - upd_names
                    if new_in_top:
                        st.markdown("**Newly surfaced**")
                        for s in updated_skills:
                            if s["name"] in new_in_top:
                                st.markdown(
                                    f'<span class="dim-tag" style="background:#f9c74f;'
                                    f'color:#111">+ {s["display_name"]}</span>',
                                    unsafe_allow_html=True)
                    if dropped:
                        st.markdown("**Pushed out of top K**")
                        for s in skills:
                            if s["name"] in dropped:
                                st.markdown(
                                    f'<span class="dim-tag" style="background:#444;'
                                    f'color:#aaa">- {s["display_name"]}</span>',
                                    unsafe_allow_html=True)
                    if not new_in_top and not dropped:
                        st.info("Top predictions unchanged.")

                st.caption(
                    "Switch to Career Advisor and search for the skill you added — "
                    "this occupation will now appear in the results."
                )

        if is_cold:
            st.divider()
            sim_col, base_col = st.columns([3, 2])
            with sim_col:
                st.subheader("Most similar existing roles")
                similar = find_similar_roles(cold_text, 5, arts)
                for i, r in enumerate(similar, 1):
                    pct = int(100 * r["similarity"])
                    st.markdown(f"**{i}. {r['title']}**")
                    st.markdown(
                        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;'
                        f'background:#a78bfa"></div></div>'
                        f'<small style="color:#888">similarity {r["similarity"]:.2f}</small>',
                        unsafe_allow_html=True)
            with base_col:
                st.subheader("Cost to add a new occupation")
                st.markdown(f"""
| Method | Can predict? | Retraining needed |
|---|---|---|
| Cosine CF | No | Full rebuild |
| Jaccard CF | No | Full rebuild |
| Node2Vec | No | Hours on GPU |
| **GraphSAGE** | **Yes** | **None — ~{elapsed_ms:.0f} ms** |
""")


# ── Mode 2: Career Advisor ────────────────────────────────────────────────────
else:
    st.header("Career Advisor")
    st.write("Select skills you have and we will find the best-matching occupations.")

    arts = get_artifacts()
    skill_names = arts["skill_names"]

    # Group by dimension
    dims: dict[str, list[str]] = {}
    for s in skill_names:
        dims.setdefault(_skill_dimension(s), []).append(s)

    top_k_roles = st.slider("Number of roles to show", 3, 15, 6)

    st.markdown("**Your skills** (pick any combination across dimensions)")
    tabs = st.tabs([DIM_LABELS.get(d, d) for d in dims])
    selected: list[str] = []
    for tab, (dim, skills_in_dim) in zip(tabs, dims.items()):
        with tab:
            chosen = st.multiselect(
                f"Select {DIM_LABELS.get(dim, dim)}",
                options=skills_in_dim,
                format_func=_skill_display,
                label_visibility="collapsed",
                key=f"tab_{dim}",
            )
            selected.extend(chosen)

    if selected:
        st.caption(f"{len(selected)} skill(s) selected: "
                   + ", ".join(_skill_display(s) for s in selected))

        # Show active session updates banner
        updates = st.session_state.get("graph_updates", {})
        if updates:
            occ_title_map = arts["maps"].get("occ_title", {})
            update_lines = []
            for oi, snames in updates.items():
                code = arts["maps"]["idx2occ"][oi]
                title = occ_title_map.get(code, code)
                update_lines.append(
                    f"**{title}** ← {', '.join(_skill_display(s) for s in snames)}"
                )
            st.warning(
                "Session graph updates active — these injections are reflected below:\n\n"
                + "\n\n".join(update_lines)
            )

        st.divider()

        with st.spinner("Finding best roles..."):
            roles = recommend_roles(selected, top_k_roles, arts, updates=updates or None)

        if not roles:
            st.warning("No results. Try selecting more skills.")
        else:
            for i, r in enumerate(roles, 1):
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        label = f"**#{i} — {r['title']}**"
                        if r.get("injected"):
                            label += "  `[graph updated this session]`"
                        st.markdown(label)
                        st.caption(r["code"])
                        bar_color = "#f9c74f" if r.get("injected") else "#e94560"
                        st.markdown(
                            f'<div class="bar-bg"><div class="bar-fill" '
                            f'style="width:{r["pct"]}%;background:{bar_color}"></div></div>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        if st.button("View skills", key=f"view_{i}"):
                            st.session_state["ca_idx"] = r["occ_idx"]
                            st.session_state["ca_title"] = r["title"]

            # Always surface graph-updated occupations even if outside top-K
            if updates:
                top_idxs = {r["occ_idx"] for r in roles}
                occ_title_map_ca = arts["maps"].get("occ_title", {})
                extra = [(oi, snames) for oi, snames in updates.items()
                         if oi not in top_idxs]
                if extra:
                    st.divider()
                    st.markdown("**Graph-updated roles** (outside top results, included because of session injection):")
                    for oi, snames in extra:
                        code = arts["maps"]["idx2occ"][oi]
                        title = occ_title_map_ca.get(code, code)
                        added_display = ", ".join(_skill_display(s) for s in snames)
                        with st.container(border=True):
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.markdown(f"**{title}**  `[graph updated — added: {added_display}]`")
                                st.caption(code)
                                st.markdown(
                                    '<div class="bar-bg"><div class="bar-fill" '
                                    'style="width:30%;background:#f9c74f"></div></div>',
                                    unsafe_allow_html=True)
                            with c2:
                                if st.button("View skills", key=f"view_inj_{oi}"):
                                    st.session_state["ca_idx"] = oi
                                    st.session_state["ca_title"] = title

        # Inline skill graph when user clicks View skills
        if "ca_idx" in st.session_state:
            st.divider()
            title = st.session_state["ca_title"]
            idx = st.session_state["ca_idx"]
            st.subheader(f"Skills for: {title}")
            with st.spinner("Loading..."):
                skills, _, _ = predict_skills(title, "", 12, arts, existing_occ_idx=idx)
            g_col, l_col = st.columns([3, 2])
            with g_col:
                fig = draw_graph(title, skills, height=400)
                st.plotly_chart(fig, width="stretch")
            with l_col:
                st.markdown("**Top skills**")
                for s in skills[:10]:
                    st.markdown(
                        f'<span class="dim-tag" style="background:{s["color"]};'
                        f'color:#111">{s["display_name"]}</span>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Select skills from the tabs above to get role recommendations.")
