"""Streamlit UI: retrieval-only by default over this repo's two-source
corpus, with optional LLM generation gated behind a user-supplied API key.

Run: streamlit run app/streamlit_app.py

Thin layer over app_logic.py -- this file is UI wiring only; the query
logic it calls is the same code tests/test_app_logic.py exercises
directly, with no Streamlit import required for those tests.
"""
import sys
from pathlib import Path

import streamlit as st

# Explicit, not relying on the runner to add this script's directory to
# sys.path -- `streamlit run` and streamlit.testing.v1.AppTest execute
# this file differently (AppTest does not add it automatically), so this
# import must not depend on which one is in play.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_logic import PRESET_QUERIES, generate_answer, load_app_vectorstore, run_query  # noqa: E402

st.set_page_config(page_title="RAG Ingestion Evaluation", page_icon="🔍")

st.title("RAG Ingestion Evaluation")
st.caption(
    "Retrieval over two public corpora: Poshan Tracker app FAQs and the "
    "Mission Saksham Anganwadi & Poshan 2.0 scheme guidelines. "
    "Retrieval-only by default -- no API key or LLM call required."
)


@st.cache_resource(show_spinner="Loading the index...")
def get_vectorstore():
    # Loads the prebuilt, committed index (app/prebuilt_index/) with ONNX
    # query embeddings -- no torch, no re-embedding the corpus at
    # startup. See app_logic.load_app_vectorstore's docstring.
    return load_app_vectorstore()


with st.sidebar:
    st.header("Optional: generate an answer")
    api_key = st.text_input(
        "OpenAI API key (optional)",
        type="password",
        help="Used only for this session's requests. Never stored, logged, or persisted.",
    )
    st.caption(
        "⚠️ Not stored, not logged, not written to disk anywhere in this "
        "app. Held in memory for this browser session only, sent directly "
        "to OpenAI per request, and gone when you close or refresh the tab."
    )

    st.divider()
    k = st.slider(
        "Number of chunks to retrieve (k)", min_value=1, max_value=10, value=5
    )
    st.caption(
        "Try raising k on the ⚠️ preset below (a query the corpus can't "
        "answer) -- results stop improving well before k=10. That's phase "
        "2's Finding #1/#4: raising k doesn't rescue a miss caused by the "
        "embedding model's representation, not retrieval depth. Full "
        "writeup: `results/ANALYSIS.md`."
    )

st.subheader("Try an example")
preset_cols = st.columns(len(PRESET_QUERIES))
for col, preset in zip(preset_cols, PRESET_QUERIES):
    if col.button(preset["label"], use_container_width=True):
        st.session_state["query"] = preset["query"]

query = st.text_input("Ask a question", key="query")

with st.expander("What do the similarity scores mean?"):
    st.markdown(
        "Each result below shows Chroma's raw similarity **distance** -- "
        "**lower means more similar**, not a percentage or a probability, "
        "and there's no fixed scale where a particular number means "
        '"confident."\n\n'
        "**This evaluation's phase 2 finding: that score cannot be used as "
        "a confidence gate.** Across all 12 tested ingestion/chunking/"
        "retriever configurations, unanswerable queries' top-1 scores "
        "overlap the same range as answerable queries' scores -- some "
        "answerable queries score *worse* than every unanswerable one "
        "tested, and vice versa. There is no threshold that reliably "
        'separates "found it" from "found the least-bad thing available." '
        "Try the ⚠️ preset above and compare its score to the others -- "
        "it won't obviously stand out.\n\n"
        "Full analysis, including the k=5-to-k=10 plateau this UI's k "
        "slider lets you reproduce: `results/ANALYSIS.md`."
    )

if query:
    vectorstore = get_vectorstore()
    results = run_query(vectorstore, query, k)

    if not results:
        st.info("No results.")
    else:
        if api_key:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, results, api_key)
            if answer:
                st.subheader("Generated answer")
                st.write(answer)
                st.caption(
                    "Generated from the retrieved chunks below, using the "
                    "API key you provided. Not grounded beyond what's shown."
                )

        st.subheader(f"Top {len(results)} retrieved chunk(s)")
        for i, result in enumerate(results, start=1):
            metadata = result["metadata"]
            if result["source"] == "faq":
                source_label = "📋 FAQ"
                detail = (
                    f"tab={metadata.get('tab')}, "
                    f"subcategory={metadata.get('subcategory') or '(none)'}"
                )
            else:
                source_label = "📄 Policy PDF"
                detail = f"page={metadata.get('page')}"

            st.markdown(
                f"**[{i}] {source_label}** -- {detail} -- "
                f"distance={result['score']:.4f} *(lower = more similar)*"
            )
            st.text(result["content"][:500])
else:
    st.info("Enter a question above, or click one of the examples.")
