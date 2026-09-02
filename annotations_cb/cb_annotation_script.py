import os
from pathlib import Path
import pandas as pd
import streamlit as st

# 1. Resolve CSV file path relative to this script directory
CURRENT_DIR = Path(__file__).parent
FILE_PATH = CURRENT_DIR / "commitmentbank_samples.csv"

# 2. Annotator Input
annotator_id = st.sidebar.text_input("Enter your Annotator ID / Name:").strip()

if not annotator_id:
    st.warning("Please enter your Annotator ID in the sidebar to start.")
    st.stop()

# Reset session state if the annotator ID changed
if st.session_state.get("current_annotator") != annotator_id:
    st.session_state.current_annotator = annotator_id
    if "data" in st.session_state:
        del st.session_state.data
    if "index" in st.session_state:
        del st.session_state.index

# Unique output filename per annotator
OUTPUT_PATH = CURRENT_DIR / f"annotated_samples_{annotator_id}.csv"

# 3. Load & Initialize Session State Data with Guaranteed Order
if "data" not in st.session_state:
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
    else:
        df = pd.read_csv(FILE_PATH, header=None, names=["text"])
        # Create an immutable ID based on the initial file order
        df["sample_id"] = df.index

    # Ensure required columns exist
    if "sample_id" not in df.columns:
        df["sample_id"] = df.index
    if "labels" not in df.columns:
        df["labels"] = None
    if "comments" not in df.columns:
        df["comments"] = None

    # Enforce deterministic order and continuous 0-based index
    df = df.sort_values(by="sample_id").reset_index(drop=True)

    st.session_state.data = df

total = len(st.session_state.data)

# 4. Track current index
if "index" not in st.session_state:
    unlabeled = st.session_state.data[st.session_state.data["labels"].isna()]
    st.session_state.index = int(unlabeled.index[0]) if not unlabeled.empty else 0


# 5. Jump to Sample Controls (Sidebar)
def on_jump():
    # Convert 1-based UI number to 0-based index
    st.session_state.index = st.session_state.jump_input - 1


st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
st.sidebar.number_input(
    label=f"Jump to sample (1 - {total}):",
    min_value=1,
    max_value=total,
    value=st.session_state.index + 1 if st.session_state.index < total else total,
    key="jump_input",
    on_change=on_jump,
)

# 6. Sidebar Download Button
st.sidebar.markdown("---")
csv_data = st.session_state.data.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="📥 Download Progress CSV",
    data=csv_data,
    file_name=f"annotated_samples_{annotator_id}.csv",
    mime="text/csv",
)

idx = st.session_state.index

# 7. Annotation UI
if idx < total:
    st.write(f"### Sample {idx + 1} of {total}")
    st.progress((idx + 1) / total)

    st.info(st.session_state.data.iloc[idx, 0])

    # Retrieve existing saved values for pre-fill
    saved_val = st.session_state.data.loc[idx, "labels"]
    default_selection = []
    if pd.notna(saved_val) and saved_val not in ["Skipped", "None"]:
        default_selection = [item.strip() for item in str(saved_val).split(" | ")]

    saved_comment = st.session_state.data.loc[idx, "comments"]
    default_comment = str(saved_comment) if pd.notna(saved_comment) else ""

    options = [
        "matrix factive verb",
        "matrix non-factive verb",
        "subject of the matrix verb",
        "tense of the predicate",
        "CC is embedded under a question",
        "CC is embedded under a modal verb",
        "CC is embedded under a negation",
        "CC is embedded under a conditional clause",
        "context of the clause",
    ]

    choice = st.multiselect(
        "Does the explanation refer to the following element to form its prediction? Multiple answers are possible:",
        options=options,
        default=default_selection,
        key=f"multi_{idx}",
    )

    comment = st.text_area(
        "Additional comments / notes (optional):",
        value=default_comment,
        key=f"comment_{idx}",
        height=80,
    )

    col_back, col_skip, col_next = st.columns([1, 1, 2])

    with col_back:
        if st.button("⬅️ Back", disabled=(idx == 0)):
            st.session_state.index -= 1
            st.rerun()

    with col_skip:
        if st.button("⏭️ Skip"):
            if pd.isna(st.session_state.data.loc[idx, "labels"]):
                st.session_state.data.loc[idx, "labels"] = "Skipped"
                st.session_state.data.loc[idx, "comments"] = (
                    comment if comment.strip() else None
                )
                st.session_state.data.to_csv(OUTPUT_PATH, index=False)
            st.session_state.index += 1
            st.rerun()

    with col_next:
        if st.button("💾 Save & Next", type="primary"):
            st.session_state.data.loc[idx, "labels"] = (
                " | ".join(choice) if choice else "None"
            )
            st.session_state.data.loc[idx, "comments"] = (
                comment.strip() if comment.strip() else None
            )
            st.session_state.data.to_csv(OUTPUT_PATH, index=False)
            st.session_state.index += 1
            st.rerun()

else:
    st.success("All samples processed!")
    st.dataframe(st.session_state.data)

    st.download_button(
        label="📥 Download Completed CSV",
        data=st.session_state.data.to_csv(index=False).encode("utf-8"),
        file_name=f"completed_annotations_{annotator_id}.csv",
        mime="text/csv",
        type="primary",
    )

    if st.button("Review from Start"):
        st.session_state.index = 0
        st.rerun()
