import streamlit as st
import pandas as pd
import os

FILE_PATH = "annotations_cb/commitmentbank_samples.csv"
# In your sidebar:
annotator_id = st.sidebar.text_input("Enter your Annotator ID / Name:").strip()
# In the sidebar so annotators can download their progress anytime:
csv_data = st.session_state.data.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="📥 Download Progress CSV",
    data=csv_data,
    file_name=f"annotated_samples_{annotator_id}.csv",
    mime="text/csv"
)

if not annotator_id:
    st.warning("Please enter your Annotator ID in the sidebar to start.")
    st.stop()

# Generate a unique output file per person
OUTPUT_PATH = f"annotated_samples_{annotator_id}.csv"

# 1. Load data
if "data" not in st.session_state:
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
    else:
        # header=None prevents the first sample from becoming the column name
        df = pd.read_csv(FILE_PATH, header=None, names=["text"])
    
    # Ensure necessary columns exist
    if "labels" not in df.columns:
        df["labels"] = None
    if "comments" not in df.columns:
        df["comments"] = None
        
    st.session_state.data = df

# 2. Track current index
if "index" not in st.session_state:
    unlabeled = st.session_state.data[st.session_state.data["labels"].isna()]
    st.session_state.index = unlabeled.index[0] if not unlabeled.empty else 0

idx = st.session_state.index
total = len(st.session_state.data)

# 3. Annotation UI
if idx < total:
    st.write(f"### Sample {idx + 1} of {total}")
    st.progress((idx + 1) / total)
    
    # Display the sample explanation/text
    st.info(st.session_state.data.iloc[idx, 0])

    # Retrieve existing saved values to pre-fill when navigating back
    saved_val = st.session_state.data.loc[idx, "labels"]
    default_selection = []
    if pd.notna(saved_val) and saved_val not in ["Skipped", "None"]:
        default_selection = [item.strip() for item in str(saved_val).split(" | ")]

    # Retrieve existing comment (if any)
    saved_comment = st.session_state.data.loc[idx, "comments"]
    default_comment = str(saved_comment) if pd.notna(saved_comment) else ""

    # Options definition
    options = [
        "matrix factive verb",
        "matrix non-factive verb",
        "subject of the matrix verb",
        "tense of the predicate",
        "CC is embedded under a question",
        "CC is embedded under a modal verb",
        "CC is embedded under a negation",
        "CC is embedded under a conditional clause",
        "context of the clause"
    ]

    choice = st.multiselect(
        "Does the explanation refer to the following element to form its prediction? Multiple answers are possible:",
        options=options,
        default=default_selection,
        key=f"multi_{idx}"
    )

    comment = st.text_area(
        "Additional comments / notes (optional):",
        value=default_comment,
        key=f"comment_{idx}",
        height=80
    )

    # 4. Navigation Buttons
    col_back, col_skip, col_next = st.columns([1, 1, 2])

    with col_back:
        if st.button("⬅️ Back", disabled=(idx == 0)):
            st.session_state.index -= 1
            st.rerun()

    with col_skip:
        if st.button("⏭️ Skip"):
            if pd.isna(st.session_state.data.loc[idx, "labels"]):
                st.session_state.data.loc[idx, "labels"] = "Skipped"
                st.session_state.data.loc[idx, "comments"] = comment if comment.strip() else None
                st.session_state.data.to_csv(OUTPUT_PATH, index=False)
            st.session_state.index += 1
            st.rerun()

    with col_next:
        if st.button("💾 Save & Next", type="primary"):
            st.session_state.data.loc[idx, "labels"] = " | ".join(choice) if choice else "None"
            st.session_state.data.loc[idx, "comments"] = comment.strip() if comment.strip() else None
            st.session_state.data.to_csv(OUTPUT_PATH, index=False)
            st.session_state.index += 1
            st.rerun()

else:
    st.success("All samples processed!")
    st.dataframe(st.session_state.data)
    
    # Download final completed file
    st.download_button(
        label="📥 Download Completed CSV",
        data=st.session_state.data.to_csv(index=False).encode('utf-8'),
        file_name=f"completed_annotations_{annotator_id}.csv",
        mime="text/csv",
        type="primary"
    )
    
    if st.button("Review from Start"):
        st.session_state.index = 0
        st.rerun()
