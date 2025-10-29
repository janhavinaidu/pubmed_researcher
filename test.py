import streamlit as st
import joblib
import re # Keep regex for cleaning
import numpy as np
import pandas as pd
import requests # To fetch data from PubMed API
# Import requests specific exceptions for better handling
from requests.exceptions import RequestException, Timeout, HTTPError
import xml.etree.ElementTree as ET # To parse the PubMed XML response
from concurrent.futures import ThreadPoolExecutor # For parallel API calls
import plotly.express as px # For plotting
# Removed plotly.graph_objects as go since radar chart is removed
from collections import Counter # For counting category frequencies
import time # To add slight delay for API politeness

# --- App Configuration ---
st.set_page_config(
    page_title="Author MeSH Profile Generator",
    page_icon="👤",
    layout="wide"
)

# --- NLTK Code REMOVED ---
# No NLTK imports or downloads needed

# --- Data & Model Loading ---
@st.cache_resource
def load_models():
    """Loads the final TF-IDF vectorizer and the best trained model."""
    try:
        # ***MODIFICATION: Load final model files***
        tfidf_filename = 'tfidf_vectorizer_final.joblib'
        model_filename = 'best_model_final.joblib'
        tfidf = joblib.load(tfidf_filename)
        model = joblib.load(model_filename)
        st.success(f"Loaded model '{model_filename}' and vectorizer '{tfidf_filename}'")
        return tfidf, model
    except FileNotFoundError:
        st.error(f"Model files not found! Ensure '{tfidf_filename}' and '{model_filename}' are present.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model files: {e}. Ensure scikit-learn version matches.")
        st.stop()

tfidf_vectorizer, model = load_models()

# --- Static Information ---
MESH_ROOT_DEFINITIONS = {
    'A': 'Anatomy', 'B': 'Organisms', 'C': 'Diseases', 'D': 'Chemicals and Drugs',
    'E': 'Analytical, Diagnostic and Therapeutic Techniques, and Equipment',
    'F': 'Psychiatry and Psychology', 'G': 'Phenomena and Processes',
    'H': 'Disciplines and Occupations', 'I': 'Anthropology, Education, Sociology, and Social Phenomena',
    'J': 'Technology, Industry, and Agriculture', 'L': 'Information Science',
    'M': 'Named Groups', 'N': 'Health Care', 'Z': 'Geographicals'
}
ROOT_LABELS_ORDER = list(MESH_ROOT_DEFINITIONS.keys()) # Ensure consistent order

# --- Backend Functions ---
POLITENESS_DELAY = 1.1 # Seconds between EACH request attempt

@st.cache_data
def search_pubmed_pmids_author(author_query, max_results=100):
    """Searches PubMed for PMIDs by author, filtering for those with abstracts."""
    # (No changes needed in function logic, uses updated POLITENESS_DELAY)
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_term = f"({author_query}) AND hasabstract[Filter]"
        params = {"db": "pubmed", "term": search_term, "retmax": max_results, "sort": "date", "retmode": "json"}
        time.sleep(0.5)
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        count = int(data.get("esearchresult", {}).get("count", 0))
        return pmids, count
    except Timeout: st.error(f"PubMed search timed out: {author_query}"); return [], 0
    except HTTPError as e: st.error(f"PubMed search HTTP error: {e}"); return [], 0
    except RequestException as e: st.error(f"PubMed search connection error: {e}"); return [], 0
    except Exception as e: st.error(f"PubMed search unexpected error: {e}"); return [], 0


@st.cache_data
def fetch_pubmed_details(pmid, max_retries=1, retry_delay=2):
    """Fetches title and abstract for a single PMID with retries, respecting rate limits."""
    # (No changes needed in function logic, uses updated POLITENESS_DELAY)
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "abstract"}
    fetch_status = "Fetch Error"; title = f"Error fetching {pmid}"; abstract = ""
    for attempt in range(max_retries + 1):
        try:
            time.sleep(POLITENESS_DELAY)
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            title_element = root.find('.//ArticleTitle')
            abstract_element = root.find('.//AbstractText')
            title = title_element.text if title_element is not None and title_element.text else "No Title Found"
            abstract = abstract_element.text if abstract_element is not None and abstract_element.text else ""
            fetch_status = "Success"
            if not abstract or len(abstract.split()) < 5:
                 abstract = ""; fetch_status = "Abstract Empty/Too Short" if title != "No Title Found" else "Fetch Error/No Data"
            break
        except Timeout:
            fetch_status = f"Fetch Timeout (Attempt {attempt + 1})"
            if attempt < max_retries: time.sleep(retry_delay)
        except HTTPError as http_err:
            status_code = http_err.response.status_code
            fetch_status = f"HTTP Error {status_code} (Attempt {attempt + 1})"
            if status_code == 429: fetch_status = "Fetch Error 429 (Rate Limit)"; break
            elif 400 <= status_code < 500: break
            if attempt < max_retries: time.sleep(retry_delay)
        except RequestException: fetch_status = f"Connection Error (Attempt {attempt + 1})"; time.sleep(retry_delay)
        except ET.ParseError: fetch_status = "XML Parse Error"; break
        except Exception as e: fetch_status = f"Unexpected Fetch Error: {type(e).__name__}"; break
    return {"pmid": pmid, "title": title, "abstract": abstract, "status": fetch_status}


def fetch_multiple_pubmed_details(pmids):
    """Fetches details for multiple PMIDs in parallel, more slowly."""
    # (No changes needed in function logic)
    results = []; max_workers = 3
    if not isinstance(pmids, list): pmids = list(pmids)
    st.info(f"Fetching {len(pmids)} abstracts using {max_workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_pubmed_details, pmid): pmid for pmid in pmids}
        progress_bar = st.progress(0, text="Fetching abstracts...")
        total_futures = len(futures); completed_futures = 0; start_time = time.time()
        for future in futures:
            try: results.append(future.result())
            except Exception as e: pmid = futures[future]; results.append({"pmid": pmid, "title": f"Thread error for {pmid}", "abstract": "", "status": "Thread Error"})
            completed_futures += 1
            try:
                elapsed_time = time.time() - start_time
                est_remaining = (elapsed_time / completed_futures * (total_futures - completed_futures)) if completed_futures > 0 else 0
                progress_text = f"Fetching abstracts... ({completed_futures}/{total_futures}) Est: {est_remaining:.0f}s"
                progress_bar.progress(completed_futures / total_futures, text=progress_text)
            except Exception: pass
        try: progress_bar.progress(1.0, text="Fetching complete."); time.sleep(0.5); progress_bar.empty()
        except Exception: pass
    return results

# --- Simplified Preprocessing Function (NO NLTK) ---
# This MUST match the function used during final training
@st.cache_data
def preprocess_text(text):
    """Cleans text using basic string methods and regex."""
    if not isinstance(text, str): return ""
    text = text.lower() # Lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    # Stopword removal is handled by TfidfVectorizer (assuming it was trained with stop_words='english')
    # No Lemmatization
    words = text.split()
    return " ".join(words)


def predict_labels(abstract_text, _tfidf_vectorizer, _model):
    """Predicts MeSH labels using the model's .predict() method."""
    if not abstract_text: return [], "No Abstract Provided"
    # Use the simple preprocessing function defined above
    cleaned_text = preprocess_text(abstract_text)
    if not cleaned_text: return [], "Preprocessing Removed All Text"
    try:
        vectorized_text = _tfidf_vectorizer.transform([cleaned_text])
        # Check if the vectorized text is empty (e.g., only contained stop words)
        if vectorized_text.nnz == 0: # nnz = number of non-zero elements
             return [], "Text Contained Only Stop Words/Ignored Features"

        predictions = _model.predict(vectorized_text)[0]
        predicted_indices = np.where(predictions == 1)[0]
        predicted_labels = [ROOT_LABELS_ORDER[i] for i in predicted_indices]
        return predicted_labels, "Success"
    except ValueError as ve:
         if "empty vocabulary" in str(ve).lower() or "dimension mismatch" in str(ve).lower():
             # Handle cases where input might be fundamentally incompatible after vectorization
             return [], "Model Input Dimension/Vocabulary Error"
         else: return [], f"Prediction Value Error: {ve}"
    except Exception as e:
        return [], f"Prediction Error: {type(e).__name__}"


# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    "Generates a **MeSH Root Category Profile** for an author using their PubMed publications (with abstracts) and a custom ML classifier (NLTK-free)." # Updated text
)
st.sidebar.markdown("### How it Works")
st.sidebar.markdown(
    "1. Enter author name (`Fauci AS[AU]`).\n"
    "2. Fetches publications **with abstracts** from PubMed.\n"
    "3. Cleans text **without NLTK**.\n" # Updated step
    "4. Classifies abstracts via `ClassifierChain`.\n"
    "5. Displays aggregated MeSH root profile."
)
st.sidebar.markdown("### Model Performance (NLTK-Free)") # Updated heading
# ***MODIFICATION: Update these metrics based on your NLTK-free training run***
# Replace these placeholder values with the actual scores you got
nltk_free_f1 = 0.8267 # Example value from your previous output
nltk_free_hamming = 0.1295 # Example value
baseline_f1_nltk_free = 0.8068 # Example value for NB
baseline_hamming_nltk_free = 0.1580 # Example value for NB
delta_hamming = nltk_free_hamming - baseline_hamming_nltk_free

col1, col2 = st.sidebar.columns(2)
col1.metric("Weighted F1", f"{nltk_free_f1:.4f}")
col2.metric("Hamming Loss", f"{nltk_free_hamming:.4f}", delta=f"{delta_hamming:.4f} vs Baseline", delta_color="inverse")
st.sidebar.markdown("---")
with st.sidebar.expander("View MeSH Category Definitions"):
    st.json(MESH_ROOT_DEFINITIONS)

# --- Main App Interface ---
st.title("👤 Author MeSH Root Profile Generator (NLTK-Free)") # Updated title
st.write(
    "Analyze the broad thematic focus of an author's publications **that have abstracts available**. "
    "Enter the author's name in PubMed format to generate their profile."
)

# --- Input Area ---
col_input1, col_input2 = st.columns([3, 1])
with col_input1:
    author_term = st.text_input(
        "Enter Author Name (PubMed Format):",
        placeholder="e.g., Smith J[AU], Collins FS[AUTH]",
        help="Use format: LastName Initials[AU]. Add `AND Affiliation[AD]` for specificity."
    )
with col_input2:
    max_results_author = st.number_input("Max Publications (with abstracts):", min_value=10, max_value=1000, value=100, step=10)

if st.button("Generate Author Profile", type="primary", use_container_width=True):
    if author_term.strip():
        with st.spinner(f"Searching PubMed for up to {max_results_author} publications by '{author_term}' with abstracts..."):
            pmids, total_count_with_abstract = search_pubmed_pmids_author(author_term, max_results_author)

        if pmids:
            num_to_fetch = len(pmids)
            st.success(f"Found {num_to_fetch} publications with abstracts (out of {total_count_with_abstract} total matching filter). Fetching details (this may take time due to API limits)...")

            abstract_details = fetch_multiple_pubmed_details(pmids)

            # --- Processing Logic ---
            classification_results = []
            fetch_success_count = 0
            fetch_fail_count = 0
            classify_success_count = 0
            classify_fail_count = 0
            category_aggregate = Counter()
            classify_progress = st.progress(0, text="Classifying abstracts...")
            total_fetched_actual = len(abstract_details)
            processed_classify_count = 0

            for item in abstract_details:
                processed_classify_count += 1
                status = item.get("status", "Unknown Error")
                is_fetch_successful = status == "Success"

                if is_fetch_successful:
                     fetch_success_count += 1
                     predicted_labels, classify_status = predict_labels(item["abstract"], tfidf_vectorizer, model)
                     if classify_status == "Success":
                         classify_success_count += 1
                         classification_results.append({
                             "pmid": item["pmid"], "title": item["title"],
                             "predicted_labels": predicted_labels, "status": "Classified Successfully"
                         })
                         category_aggregate.update(predicted_labels)
                     else:
                         classify_fail_count += 1
                         classification_results.append({
                             "pmid": item["pmid"], "title": item["title"],
                             "predicted_labels": [], "status": classify_status
                         })
                else: # Any fetch status other than "Success" is a failure
                    fetch_fail_count += 1
                    classification_results.append({
                        "pmid": item["pmid"], "title": item["title"],
                        "predicted_labels": [], "status": status
                    })

                if total_fetched_actual > 0:
                     classify_progress.progress(processed_classify_count / total_fetched_actual, text=f"Classifying abstracts... ({processed_classify_count}/{total_fetched_actual})")

            try: classify_progress.progress(1.0, text="Classification complete."); time.sleep(0.5); classify_progress.empty()
            except Exception: pass

            st.header(f"📊 Profile for: {author_term}")

            # --- Display Summary Stats ---
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Pubs Found (w/ Abstract)", total_count_with_abstract)
            stat_col2.metric("Abstracts Attempted", num_to_fetch)
            stat_col3.metric("Successfully Classified", classify_success_count)
            combined_errors = num_to_fetch - classify_success_count
            stat_col4.metric("Fetch/Classify Errors", combined_errors)
            if combined_errors > 0:
                 error_types = Counter(res['status'] for res in classification_results if res['status'] != "Classified Successfully")
                 st.caption(f"Error breakdown:")
                 st.json(dict(error_types))


            if classify_success_count > 0:
                 # --- Display Aggregate Profile ---
                st.subheader("Overall MeSH Root Category Distribution")
                st.caption("Note: 'B: Organisms' (humans, animals, cells) is commonly predicted in biomedical texts.")

                profile_data = [{"Category": MESH_ROOT_DEFINITIONS.get(cat, cat), "Code": cat, "Count": count, "Percentage": (count / classify_success_count) * 100}
                                for cat, count in category_aggregate.items()]
                profile_df = pd.DataFrame(profile_data).sort_values("Percentage", ascending=False)

                fig_bar = px.bar(profile_df, x="Category", y="Percentage", color="Percentage",
                                 text=profile_df['Percentage'].apply(lambda x: f'{x:.1f}%'),
                                 color_continuous_scale=px.colors.sequential.Viridis,
                                 title=f"Thematic Focus by MeSH Root Category ({classify_success_count} Papers)",
                                 labels={"Category": "MeSH Root Category", "Percentage": "% of Classified Papers"})
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(height=450, uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_bar, use_container_width=True)

  

                # --- *** ADDED Classification Results Table *** ---
                st.subheader("Classification Results per Publication")
                # Prepare data for the table - only successful classifications
                table_data = []
                for result in classification_results:
                    if result['status'] == "Classified Successfully":
                        table_data.append({
                            "PMID": result["pmid"],
                            "Title": result["title"][:80] + "..." if len(result["title"]) > 80 else result["title"], # Truncate long titles
                            "Predicted Categories": ", ".join(result["predicted_labels"]) if result["predicted_labels"] else "None"
                        })

                if table_data:
                    results_df = pd.DataFrame(table_data)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                else:
                    # This case should be covered by classify_success_count > 0, but added for safety
                    st.info("No abstracts were successfully classified to display in table.")
                # --- End of Added Table Section ---


                # --- Display List of Papers Analyzed (Reduced Detail) ---
                with st.expander(f"View Fetch/Classification Status for all {len(classification_results)} Attempted Publications"):
                    st.caption("Lists all publications attempted, showing final status.")
                    for i, result in enumerate(classification_results):
                         st.markdown(f"---")
                         st.markdown(f"**{i+1}. {result['title']}** (PMID: {result['pmid']})")
                         # Show status clearly
                         if result['status'] == "Classified Successfully":
                             st.success(f"**Status:** {result['status']} (Categories: {', '.join(result['predicted_labels']) if result['predicted_labels'] else 'None Predicted'})")
                         else:
                             st.warning(f"**Status:** {result['status']}") # Show specific error
                         st.link_button("View on PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{result['pmid']}/")

            else:
                 if num_to_fetch > 0:
                     st.warning(f"Could not successfully classify any of the {num_to_fetch} fetched abstracts. Check error breakdown above.")
                 else:
                     st.warning(f"No abstracts were successfully fetched or classified for '{author_term}'.")

        else:
             if author_term:
                 st.error(f"No publications with abstracts found on PubMed matching the query '{author_term}'. Check format (e.g., 'Smith J[AU]') and spelling, or try removing filters.")
    else:
        st.warning("Please enter an author name in the specified format.")

