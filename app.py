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
client = openai.OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
)

# Custom headers for requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; WaybackMachineClient/1.0; +http://yourdomain.com)'
}

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=False
)
def get_wayback_snapshots(url, target_date, match_type='exact', filters=None, collapse=None, limit=1):
    """
    Fetch snapshots for a given URL and date from the Wayback CDX Server API.

    Parameters:
        url (str): The URL to query.
        target_date (str): The date for the snapshot (format: 'YYYY-MM-DD' or 'MM/DD/YYYY').
        match_type (str): The match type ('exact', 'prefix', 'host', 'domain').
        filters (list): List of filter strings, e.g., ['statuscode:200', '!mimetype:image/png'].
        collapse (list): List of fields to collapse results on, e.g., ['digest', 'timestamp:10'].
        limit (int): Number of results to fetch.

    Returns:
        tuple: (list of snapshot dictionaries, error message or None)
    """
    try:
        # Parse and format the date
        date_obj = pd.to_datetime(target_date, errors='coerce')
        if pd.isnull(date_obj):
            error_msg = f"Invalid date format: '{target_date}'. Use 'YYYY-MM-DD' or 'MM/DD/YYYY'."
            logger.warning(error_msg)
            st.warning(error_msg)
            return [], error_msg

        date_str = date_obj.strftime("%Y%m%d")  # Format: YYYYMMDD

        # Construct the base CDX API URL
        cdx_url = "https://web.archive.org/cdx/search/cdx"

        # Prepare query parameters
        params = {
            'url': url,
            'output': 'json',
            'from': date_str,
            'to': date_str,
            'limit': limit,
            'matchType': match_type
        }

        # Add filters if provided
        if filters:
            for filt in filters:
                params.setdefault('filter', []).append(filt)

        # Add collapse parameters if provided
        if collapse:
            for col in collapse:
                params.setdefault('collapse', []).append(col)

        # Construct the final query string with multiple filters and collapse parameters
        query_params = []
        for key, value in params.items():
            if isinstance(value, list):
                for item in value:
                    query_params.append(f"{key}={item}")
            else:
                query_params.append(f"{key}={value}")
        query_string = '&'.join(query_params)
        full_url = f"{cdx_url}?{query_string}"

        logger.info(f"Querying CDX API: {full_url}")

        # Make the request with custom headers
        response = requests.get(full_url, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:
                # The first row is the header
                headers = data[0]
                snapshots = [dict(zip(headers, row)) for row in data[1:]]
                logger.info(f"Found {len(snapshots)} snapshots for {url} on {target_date}.")
                return snapshots, None
            else:
                warning_msg = f"No snapshot found for {url} on {target_date}."
                logger.warning(warning_msg)
                st.warning(warning_msg)
                return [], warning_msg
        else:
            warning_msg = f"Failed to fetch snapshot for {url} on {target_date}. Status Code: {response.status_code}"
            logger.warning(warning_msg)
            st.warning(warning_msg)
            return [], warning_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"An error occurred while fetching snapshot for {url} on {target_date}: {e}"
        logger.error(error_msg)
        st.warning(error_msg)
        return [], error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logger.error(error_msg)
        st.warning(error_msg)
        return [], error_msg

def get_oldest_snapshot(url):
    """
    Fetch the oldest snapshot available for a given URL from the Wayback CDX Server API.

    Parameters:
        url (str): The URL to query.

    Returns:
        tuple: (snapshot dictionary, error message or None)
    """
    try:
        # No date range specified to fetch the oldest snapshot
        cdx_url = "https://web.archive.org/cdx/search/cdx"
        params = {
            'url': url,
            'output': 'json',
            'limit': 1,
            'sort': 'ascending',  # Oldest first
            'matchType': 'exact'
        }

        # Construct the query string
        query_params = [f"{key}={value}" for key, value in params.items()]
        query_string = '&'.join(query_params)
        full_url = f"{cdx_url}?{query_string}"

        logger.info(f"Fetching oldest snapshot: {full_url}")

        # Make the request with custom headers
        response = requests.get(full_url, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:
                headers = data[0]
                snapshot = dict(zip(headers, data[1]))
                logger.info(f"Oldest snapshot found: {snapshot}")
                return snapshot, None
            else:
                warning_msg = f"No snapshots available for {url}."
                logger.warning(warning_msg)
                st.warning(warning_msg)
                return None, warning_msg
        else:
            warning_msg = f"Failed to fetch oldest snapshot for {url}. Status Code: {response.status_code}"
            logger.warning(warning_msg)
            st.warning(warning_msg)
            return None, warning_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"An error occurred while fetching the oldest snapshot for {url}: {e}"
        logger.error(error_msg)
        st.warning(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while fetching the oldest snapshot for {url}: {e}"
        logger.error(error_msg)
        st.warning(error_msg)
        return None, error_msg

def fetch_content_from_snapshot(archived_url):
    """
    Fetch and extract textual content from the archived snapshot URL.
    """
    try:
        # Make the request with custom headers
        response = requests.get(archived_url, headers=HEADERS, timeout=10)

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
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Assess how much the following content has changed since the prior date. "
                        "Evaluate if it is a small, medium, or large change to the content or if it is a complete overhaul of the content. "
                        "Return a score of Small, Medium, Large, or Overhaul. This should be the only thing returned.\n\n"
                        f"Before Content:\n{before_content}\n\n"
                        f"After Content:\n{after_content}"
                    ),
                }
            ],
            model="gpt-4o-mini",
        )

        # Extract the response text
        evaluation = chat_completion.choices[0].message.content.strip()
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

                    # Initialize results list
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total = len(df)

                    for idx, row in df.iterrows():
                        url = row[url_col]
                        before_date = row[before_date_col]
                        after_date = row[after_date_col]

                        status_text.text(f"Processing row {idx + 1} of {total}: {url}")

                        # Initialize error and oldest_snapshot_date
                        error = None
                        oldest_snapshot_date = None

                        # Fetch before snapshots
                        before_snapshots, before_error = get_wayback_snapshots(
                            url=url,
                            target_date=before_date,
                            match_type=match_type,
                            filters=filters,
                            collapse=collapse,
                            limit=limit
                        )

                        if before_error:
                            error = before_error
                            # Attempt to fetch the oldest snapshot
                            oldest_snapshot, oldest_error = get_oldest_snapshot(url)
                            if oldest_snapshot:
                                oldest_snapshot_date = pd.to_datetime(oldest_snapshot['timestamp'], format="%Y%m%d%H%M%S").strftime("%Y-%m-%d")
                                before_content = fetch_content_from_snapshot(oldest_snapshot['original'])
                                if not before_content:
                                    error += " Failed to extract 'Before' content from the oldest snapshot."
                            else:
                                before_content = None
                                # Error message already captured
                        else:
                            if before_snapshots:
                                before_content = fetch_content_from_snapshot(before_snapshots[0]['original'])
                                if not before_content:
                                    error = "Failed to extract 'Before' content."
                            else:
                                before_content = None
                                error = "No 'Before' snapshot found."

                        # Fetch after snapshots
                        after_snapshots, after_error = get_wayback_snapshots(
                            url=url,
                            target_date=after_date,
                            match_type=match_type,
                            filters=filters,
                            collapse=collapse,
                            limit=limit
                        )

                        if after_error:
                            if error:
                                error += " | " + after_error
                            else:
                                error = after_error
                            # Attempt to fetch the oldest snapshot
                            oldest_snapshot, oldest_error = get_oldest_snapshot(url)
                            if oldest_snapshot:
                                oldest_snapshot_date = pd.to_datetime(oldest_snapshot['timestamp'], format="%Y%m%d%H%M%S").strftime("%Y-%m-%d")
                                after_content = fetch_content_from_snapshot(oldest_snapshot['original'])
                                if not after_content:
                                    error += " Failed to extract 'After' content from the oldest snapshot."
                            else:
                                after_content = None
                                # Error message already captured
                        else:
                            if after_snapshots:
                                after_content = fetch_content_from_snapshot(after_snapshots[0]['original'])
                                if not after_content:
                                    if error:
                                        error += " | Failed to extract 'After' content."
                                    else:
                                        error = "Failed to extract 'After' content."
                            else:
                                after_content = None
                                if error:
                                    error += " | No 'After' snapshot found."
                                else:
                                    error = "No 'After' snapshot found."

                        # Determine evaluation
                        if before_content and after_content:
                            evaluation = evaluate_changes(before_content, after_content)
                        else:
                            evaluation = "Insufficient Data"

                        # Append results
                        results.append({
                            "URL": url,
                            "Before Date": before_date,
                            "After Date": after_date,
                            "Oldest Snapshot Date": oldest_snapshot_date,
                            "Change Evaluation": evaluation,
                            "Error": error
                        })

                        # Update progress bar
                        progress_bar.progress((idx + 1) / total)

                    # Finalize
                    progress_bar.empty()
                    status_text.empty()
                    result_df = pd.DataFrame(results)

                    # Reorder columns for clarity
                    result_df = result_df[[
                        "URL",
                        "Before Date",
                        "After Date",
                        "Oldest Snapshot Date",
                        "Change Evaluation",
                        "Error"
                    ]]

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
