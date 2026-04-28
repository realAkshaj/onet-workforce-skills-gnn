"""Workforce Intelligence Dashboard.

Run:
    cd project
    C:\\Python313\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

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
               injected: set[str] | None = None) -> str:
    from pyvis.network import Network
    net = Network(height=f"{height}px", width="100%",
                  bgcolor="#0e1117", font_color="#e0e0e0", directed=False)
    net.set_options("""{
      "physics": {"barnesHut": {"gravitationalConstant":-8000,"springLength":160}},
      "edges": {"smooth": {"type":"continuous"}},
      "interaction": {"hover":true}
    }""")
    net.add_node(occ_label, label=occ_label, size=34, color="#e94560",
                 font={"size":14, "bold":True}, title=occ_label)
    mx = max((s["score"] for s in skills), default=1.0)
    for s in skills:
        norm = s["score"] / mx if mx > 0 else 0.5
        is_injected = injected and s["name"] in injected
        color  = "#f9c74f" if is_injected else s["color"]  # gold for injected
        label  = f"+ {s['display_name']}" if is_injected else s["display_name"]
        tip    = ("ADDED — " if is_injected else "") + \
                 f"{s['dim_label']}: {s['display_name']}  score={s['score']:.3f}"
        net.add_node(s["name"], label=label,
                     size=10 + 18*norm, color=color,
                     borderWidth=3 if is_injected else 1,
                     title=tip)
        net.add_edge(occ_label, s["name"], width=1+3*norm,
                     color={"color": "#f9c74f" if is_injected else s["color"],
                            "opacity": 0.9 if is_injected else 0.75},
                     dashes=is_injected)
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                     mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)
    return html


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
    st.caption("O*NET Release 30.1 · 894 occupations · 160 features")


# ── Mode 1: Skill Explorer ────────────────────────────────────────────────────
if mode == "Skill Explorer":
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
        new_title = st.text_input("Occupation title",
                                  placeholder="e.g. AI Policy Analyst")
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

        # Resolve existing occupation index
        existing_idx = None
        if not is_new:
            title2code = {v: k for k, v in occ_title_map.items()}
            code = title2code.get(selected_title)
            if code:
                existing_idx = arts["maps"]["occ2idx"].get(code)

        import time
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

        if is_cold:
            # Show the inductive advantage: trained once, scores new nodes instantly
            ckpt = ROOT / "checkpoints" / "sage_coldstart.pt"
            import datetime, os
            trained_on = datetime.datetime.fromtimestamp(
                os.path.getmtime(ckpt)).strftime("%b %d %Y") if ckpt.exists() else "N/A"

            m1, m2, m3 = st.columns(3)
            m1.metric("Inference time", f"{elapsed_ms:.0f} ms",
                      help="Time to score this unseen occupation")
            m2.metric("Model retrained?", "No",
                      help="Same checkpoint used for all predictions")
            m3.metric("Checkpoint trained", trained_on,
                      help="Model has not been updated since this date")
            st.info(
                f"'{occ_label}' was not in training data. "
                "GraphSAGE scores it in a single forward pass — no retraining required."
            )

        graph_col, list_col = st.columns([3, 2])

        with graph_col:
            st.subheader("Knowledge graph")
            html = draw_graph(occ_label, skills)
            st.components.v1.html(html, height=480)

        with list_col:
            st.subheader("Predicted skills")
            mx = max(s["score"] for s in skills)
            for i, s in enumerate(skills, 1):
                pct = int(100 * s["score"] / mx) if mx > 0 else 50
                col_a, col_b = st.columns([5, 2])
                with col_a:
                    st.markdown(f"**{i}. {s['display_name']}**")
                with col_b:
                    st.markdown(
                        f'<span class="dim-tag" style="background:{s["color"]};color:#111">'
                        f'{s["dim_label"]}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;'
                    f'background:{s["color"]}"></div></div>',
                    unsafe_allow_html=True,
                )

        # ── Add a skill to an existing occupation ────────────────────────────
        if not is_cold and existing_idx is not None:
            st.divider()
            st.subheader("Add a skill to this role")
            st.caption(
                "Inject a new skill into the graph and re-run inference in one forward pass. "
                "Model parameters are not changed."
            )

            # Skills not already in this occupation's training profile
            known = set(int(i) for i in arts["train_mask"][existing_idx].nonzero()[0])
            available = [arts["skill_names"][i] for i in range(arts["n_skill"])
                         if i not in known]

            # Persist additions across reruns in session state
            state_key = f"added_{existing_idx}"
            if state_key not in st.session_state:
                st.session_state[state_key] = []

            added = st.multiselect(
                "Select skills to add",
                options=available,
                default=st.session_state[state_key],
                format_func=_skill_display,
                key=f"add_ms_{existing_idx}",
            )
            st.session_state[state_key] = added

            # Persist in global updates so Career Advisor reflects this change
            if "graph_updates" not in st.session_state:
                st.session_state["graph_updates"] = {}
            if added:
                st.session_state["graph_updates"][existing_idx] = added
            elif existing_idx in st.session_state["graph_updates"]:
                del st.session_state["graph_updates"][existing_idx]

            if added:
                with st.spinner("Updating graph..."):
                    updated_skills, update_ms = predict_skills_with_update(
                        existing_idx, added, top_k, arts
                    )

                u1, u2, u3 = st.columns(3)
                u1.metric("Update time", f"{update_ms:.0f} ms")
                u2.metric("Skills added", len(added))
                u3.metric("Model retrained?", "No")

                orig_names = {s["name"] for s in skills}
                upd_names  = {s["name"] for s in updated_skills}
                new_in_top = upd_names - orig_names
                dropped    = orig_names - upd_names

                g_col, diff_col = st.columns([3, 2])
                with g_col:
                    st.subheader("Updated knowledge graph")
                    injected_set = set(added)
                    # Build display list: updated predictions + injected skills
                    graph_skills = list(updated_skills)
                    for skill_name in added:
                        dim = _skill_dimension(skill_name)
                        graph_skills.append({
                            "name": skill_name,
                            "display_name": _skill_display(skill_name),
                            "dimension": dim,
                            "dim_label": DIM_LABELS.get(dim, dim),
                            "color": "#f9c74f",
                            "score": 1.0,
                        })
                    html = draw_graph(occ_label, graph_skills,
                                      height=420, injected=injected_set)
                    st.components.v1.html(html, height=440)

                with diff_col:
                    st.subheader("What changed")
                    if new_in_top:
                        st.markdown("**Newly surfaced predictions**")
                        for s in updated_skills:
                            if s["name"] in new_in_top:
                                st.markdown(
                                    f'<span class="dim-tag" style="background:#f9c74f;'
                                    f'color:#111">+ {s["display_name"]}</span>',
                                    unsafe_allow_html=True,
                                )
                    if dropped:
                        st.markdown("**Pushed out of top K**")
                        for s in skills:
                            if s["name"] in dropped:
                                st.markdown(
                                    f'<span class="dim-tag" style="background:#444;'
                                    f'color:#aaa">- {s["display_name"]}</span>',
                                    unsafe_allow_html=True,
                                )
                    if not new_in_top and not dropped:
                        st.info("Top predictions unchanged — added skill already well-represented.")

        if is_cold:
            st.divider()
            sim_col, base_col = st.columns([3, 2])

            with sim_col:
                st.subheader("Most similar existing roles")
                st.caption("Ranked by embedding cosine similarity — the GNN places the new occupation near these in the learned space.")
                similar = find_similar_roles(cold_text, 5, arts)
                for i, r in enumerate(similar, 1):
                    pct = int(100 * r["similarity"])
                    st.markdown(f"**{i}. {r['title']}**")
                    st.markdown(
                        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;'
                        f'background:#a78bfa"></div></div>'
                        f'<small style="color:#888">similarity {r["similarity"]:.2f}</small>',
                        unsafe_allow_html=True,
                    )

            with base_col:
                st.subheader("Cost to add a new occupation")
                st.markdown("""
| Method | Can predict? | Retraining needed |
|---|---|---|
| Cosine CF | No | Full rebuild |
| Jaccard CF | No | Full rebuild |
| Node2Vec | No | Hours on GPU |
| **GraphSAGE** | **Yes** | **None — ~{:.0f} ms** |
""".format(elapsed_ms), unsafe_allow_html=False)


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
                html = draw_graph(title, skills, height=400)
                st.components.v1.html(html, height=420)
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
