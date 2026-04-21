import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from utils.code_extractor import extract_code_from_file
from utils.gnn_inference import load_gnn_model, compute_code_similarity

st.set_page_config(page_title="Code Similarity Suite", layout="wide")
st.title("🔍 Code Similarity Analysis Suite")

tab1, tab2 = st.tabs(["🧠 CodeGNN Clone Detection (C++)", "📋 Assignment Code Checker"])

# ===================== TAB 1: CodeGNN Pairwise =====================
with tab1:
    st.header("CodeGNN Clone Detection (C/C++)")
    st.markdown("Paste two C++ code snippets and get semantic similarity using Graph Neural Network (AST + CFG + PDG).")

    col1, col2 = st.columns(2)
    with col1:
        code1 = st.text_area("Code 1 (C++)", height=400, placeholder="// Paste first C++ code here...")
    with col2:
        code2 = st.text_area("Code 2 (C++)", height=400, placeholder="// Paste second C++ code here...")

    if st.button("🔬 Compare with CodeGNN", type="primary", use_container_width=True):
        if not code1.strip() or not code2.strip():
            st.error("Please enter both codes.")
        else:
            with st.spinner("Loading model and computing embeddings..."):
                model, device = load_gnn_model()
                similarity = compute_code_similarity(model, code1, code2, device)
                
                if isinstance(similarity, (int, float)):
                    st.success(f"**Similarity Score: {similarity}%**")
                    if similarity > 85:
                        st.error("🔴 Very High Similarity - Possible Clone")
                    elif similarity > 70:
                        st.warning("🟡 High Similarity")
                    else:
                        st.info("🟢 Low Similarity")
                else:
                    st.error(similarity)

# ===================== TAB 2: Multiple Files Checker =====================
with tab2:
    st.header("📋 Assignment Code Similarity Checker")
    st.markdown("**For Teachers** • Upload multiple student submissions → Get similarity heatmap")

    uploaded_files = st.file_uploader(
        "Upload student assignments (PDF, DOCX, TXT, .py, .java, .cpp, etc.)",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Analyze All Submissions", type="primary"):
        if len(uploaded_files) < 2:
            st.error("Upload at least 2 files.")
        else:
            with st.spinner("Extracting code and computing similarities..."):
                from copydetect import CodeFingerprint, compare_files
                import io

                student_codes = {}
                for file in uploaded_files:
                    code = extract_code_from_file(file)
                    if code.strip():
                        student_codes[file.name] = code

                if len(student_codes) < 2:
                    st.error("Not enough valid code found.")
                    st.stop()

                names = list(student_codes.keys())
                similarity_matrix = pd.DataFrame(0.0, index=names, columns=names)
                results = []

                for i in range(len(names)):
                    for j in range(i+1, len(names)):
                        name1, name2 = names[i], names[j]
                        code1, code2 = student_codes[name1], student_codes[name2]

                        try:
                            fp1 = CodeFingerprint(file=io.StringIO(code1), k=25, win_size=1)
                            fp2 = CodeFingerprint(file=io.StringIO(code2), k=25, win_size=1)
                            _, sims, _ = compare_files(fp1, fp2)
                            sim_score = max(sims)

                            similarity_matrix.loc[name1, name2] = sim_score
                            similarity_matrix.loc[name2, name1] = sim_score

                            results.append({
                                "Student 1": name1,
                                "Student 2": name2,
                                "Similarity": f"{sim_score:.1%}"
                            })
                        except:
                            pass

                # Heatmap
                fig = px.imshow(similarity_matrix.values, x=names, y=names, 
                               text_auto=".1%", color_continuous_scale="RdYlGn_r")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Results table
                st.subheader("Pairwise Results")
                df = pd.DataFrame(results)
                if not df.empty:
                    df = df.sort_values("Similarity", ascending=False)
                    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("CodeGNN Tab uses custom Graph Neural Network • Multiple Checker uses winnowing fingerprints")
