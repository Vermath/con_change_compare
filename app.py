import streamlit as st
import pandas as pd
import requests
import openai
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client using Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True
)
def get_wayback_snapshots(url, target_date, match_type='exact', filters=None, collapse=None, limit=1):
    """
    Fetch snapshots for a given URL and date from the Wayback CDX Server API.
    
    Parameters:
        url (str): The URL to query.
        target_date (str): The date for the snapshot (format: 'YYYY-MM-DD' or 'MM/DD/YYYY').
        match_type (str): The match type ('exact', 'prefix', 'host', 'domain').
        filters (list): List of filter strings, e.g., ['statuscode:200', '!mimetype:image/png'].
        collapse (str): The field to collapse results on, e.g., 'digest', 'timestamp:10'.
        limit (int): Number of results to fetch.
        
    Returns:
        list: List of snapshot dictionaries with fields ['urlkey', 'timestamp', 'original', 'mimetype', 'statuscode', 'digest', 'length'].
    """
    try:
        # Parse and format the date
        date_obj = pd.to_datetime(target_date, errors='coerce')
        if pd.isnull(date_obj):
            st.warning(f"Invalid date format for '{target_date}'. Please use 'YYYY-MM-DD' or 'MM/DD/YYYY'.")
            return []
        
        date_str = date_obj.strftime("%Y%m%d")  # Format: YYYYMMDD
        
        # Construct the base CDX API URL
        cdx_url = "http://web.archive.org/cdx/search/cdx"
        
        # Prepare query parameters
        params = {
            'url': url,
            'output': 'json',
            'from': date_str,
            'to': date_str,
            'limit': limit,
            'filter': filters,
            'collapse': collapse,
            'matchType': match_type
        }
        
        # Remove None or empty parameters
        params = {k: v for k, v in params.items() if v}
        
        # For filters, if it's a list, join them with '&filter='
        # e.g., ['statuscode:200', '!mimetype:image/png'] becomes 'filter=statuscode:200&filter=!mimetype:image/png'
        if 'filter' in params and isinstance(params['filter'], list):
            filter_params = '&'.join([f'filter={filt}' for filt in params['filter']])
            # Remove 'filter' key and append filter_params to the URL
            del params['filter']
        else:
            filter_params = ''
        
        # Similarly handle collapse if multiple
        if 'collapse' in params and isinstance(params['collapse'], list):
            collapse_params = '&'.join([f'collapse={col}' for col in params['collapse']])
            del params['collapse']
        else:
            collapse_params = ''
        
        # Construct the final query string
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        if filter_params:
            query_string += f"&{filter_params}"
        if collapse_params:
            query_string += f"&{collapse_params}"
        
        full_url = f"{cdx_url}?{query_string}"
        
        logger.info(f"Querying CDX API: {full_url}")
        
        response = requests.get(full_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:
                # The first row is the header
                headers = data[0]
                snapshots = [dict(zip(headers, row)) for row in data[1:]]
                logger.info(f"Found {len(snapshots)} snapshots for {url} on {target_date}.")
                return snapshots
            else:
                st.warning(f"No snapshot found for {url} on {target_date}.")
                return []
        elif response.status_code == 503:
            # Raise exception to trigger retry
            response.raise_for_status()
        else:
            st.warning(f"Failed to fetch snapshot for {url} on {target_date}. Status Code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException for {url} on {target_date}: {e}")
        st.warning(f"An error occurred while fetching snapshot for {url} on {target_date}: {e}")
        raise  # Trigger retry
    except Exception as e:
        logger.error(f"Unexpected error for {url} on {target_date}: {e}")
        st.warning(f"An unexpected error occurred: {e}")
        return []

def fetch_content_from_snapshot(archived_url):
    """
    Fetch and extract textual content from the archived snapshot URL.
    """
    try:
        response = requests.get(archived_url, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n')
            
            # Collapse multiple newlines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        else:
            logger.warning(f"Failed to fetch content from {archived_url}. Status Code: {response.status_code}")
            st.warning(f"Failed to fetch content from {archived_url}. Status Code: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"An error occurred while fetching content from {archived_url}: {e}")
        st.warning(f"An error occurred while fetching content from {archived_url}: {e}")
        return None

def evaluate_changes(before_content, after_content):
    """
    Use OpenAI API to evaluate the changes between two pieces of content.
    Returns one of: Small, Medium, Large, Overhaul
    """
    try:
        prompt = (
            "Assess how much the following content has changed since the prior date. "
            "Evaluate if it is a small, medium, or large change to the content or if it is a complete overhaul of the content. "
            "Return a score of Small, Medium, Large, or Overhaul. This should be the only thing returned."
        )
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"Before Content:\n{before_content}\n\n"
                    f"After Content:\n{after_content}"
                ),
            }
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        # Extract the response text
        evaluation = response.choices[0].message['content'].strip()
        return evaluation
    except Exception as e:
        logger.error(f"An error occurred while evaluating changes: {e}")
        st.error(f"An error occurred while evaluating changes: {e}")
        return "Error"

def main():
    st.title("Website Content Change Evaluator Using Wayback CDX Server API")
    st.write("""
        Upload a CSV file containing URLs, before dates, and after dates.
        Define which columns correspond to each field.
        The app will fetch the content from each URL on the specified dates using the Wayback CDX Server API and evaluate the changes.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")
            
            # Display the first few rows
            st.dataframe(df.head())

            # Let user map columns
            with st.form("column_mapping"):
                st.header("Define Column Mapping")
                url_col = st.selectbox("Select the URL column", options=df.columns)
                before_date_col = st.selectbox("Select the Before Date column (e.g., '2024-10-28' or '10/28/2024')", options=df.columns)
                after_date_col = st.selectbox("Select the After Date column (e.g., '2024-11-28' or '11/28/2024')", options=df.columns)
                # Advanced Query Parameters
                st.subheader("Advanced Query Parameters (Optional)")
                match_type = st.selectbox(
                    "Select Match Type",
                    options=['exact', 'prefix', 'host', 'domain'],
                    index=0
                )
                collapse_options = st.text_input(
                    "Collapse Results (comma-separated fields, e.g., 'digest,timestamp:10')",
                    value="",
                    help="Use 'collapse=field' or 'collapse=field:N' separated by commas."
                )
                filters_input = st.text_input(
                    "Filters (comma-separated, e.g., 'statuscode:200,!mimetype:image/png')",
                    value="",
                    help="Use 'filter=field:regex'. Prefix with '!' to invert the match."
                )
                limit = st.number_input(
                    "Number of Snapshots to Fetch",
                    min_value=1,
                    max_value=100,
                    value=1,
                    step=1
                )
                submit_mapping = st.form_submit_button("Submit")

            if submit_mapping:
                # Validate that selected columns are unique
                if len({url_col, before_date_col, after_date_col}) < 3:
                    st.error("Please select three distinct columns for URL, Before Date, and After Date.")
                else:
                    st.success("Column mapping saved!")
                    
                    # Parse collapse and filters
                    collapse = [item.strip() for item in collapse_options.split(',') if item.strip()] if collapse_options else None
                    filters = [item.strip() for item in filters_input.split(',') if item.strip()] if filters_input else None
                    
                    # Process each row
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total = len(df)
                    
                    for idx, row in df.iterrows():
                        url = row[url_col]
                        before_date = row[before_date_col]
                        after_date = row[after_date_col]
                        
                        status_text.text(f"Processing row {idx + 1} of {total}: {url}")
                        
                        # Fetch before snapshots
                        before_snapshots = get_wayback_snapshots(
                            url=url,
                            target_date=before_date,
                            match_type=match_type,
                            filters=filters,
                            collapse=collapse,
                            limit=limit
                        )
                        if before_snapshots:
                            before_content = fetch_content_from_snapshot(before_snapshots[0]['original'])
                            if not before_content:
                                before_error = "Failed to extract content."
                        else:
                            before_content = None
                            before_error = "No snapshot found."
                        
                        # Fetch after snapshots
                        after_snapshots = get_wayback_snapshots(
                            url=url,
                            target_date=after_date,
                            match_type=match_type,
                            filters=filters,
                            collapse=collapse,
                            limit=limit
                        )
                        if after_snapshots:
                            after_content = fetch_content_from_snapshot(after_snapshots[0]['original'])
                            if not after_content:
                                after_error = "Failed to extract content."
                        else:
                            after_content = None
                            after_error = "No snapshot found."
                        
                        # Determine evaluation status
                        if not before_content or not after_content:
                            if not before_snapshots and not after_snapshots:
                                evaluation = "No Snapshots Found"
                            elif not before_snapshots:
                                evaluation = "No 'Before' Snapshot"
                            elif not after_snapshots:
                                evaluation = "No 'After' Snapshot"
                            else:
                                evaluation = "Insufficient Data"
                        else:
                            # Evaluate changes using OpenAI API
                            evaluation = evaluate_changes(before_content, after_content)
                        
                        results.append({
                            "URL": url,
                            "Before Date": before_date,
                            "After Date": after_date,
                            "Change Evaluation": evaluation
                        })
                        
                        progress_bar.progress((idx + 1) / total)
                    
                    progress_bar.empty()
                    status_text.empty()
                    result_df = pd.DataFrame(results)
                    st.header("Evaluation Results")
                    st.dataframe(result_df)
                    
                    # Option to download results
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='evaluation_results.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}")

if __name__ == "__main__":
    main()
