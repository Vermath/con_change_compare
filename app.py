import streamlit as st
import pandas as pd
import requests
import openai
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Initialize OpenAI client using Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_wayback_snapshot(url, target_date):
    """
    Fetch the closest snapshot timestamp for a given URL and date from the Wayback Machine.
    """
    try:
        # Format date as YYYYMMDD
        date_str = datetime.strptime(target_date, "%Y-%m-%d").strftime("%Y%m%d")
        
        cdx_url = (
            f"http://web.archive.org/cdx/search/cdx?"
            f"url={url}&output=json&from={date_str}&to={date_str}&limit=1&filter=statuscode:200&collapse=digest"
        )
        
        response = requests.get(cdx_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:
                # The first row is the header
                snapshot = data[1]
                timestamp = snapshot[1]
                archived_url = f"http://web.archive.org/web/{timestamp}/{url}"
                return archived_url
            else:
                return None  # No snapshot found
        else:
            st.warning(f"Failed to fetch snapshot for {url} on {target_date}. Status Code: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"An error occurred while fetching snapshot for {url} on {target_date}: {e}")
        return None

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
            st.warning(f"Failed to fetch content from {archived_url}. Status Code: {response.status_code}")
            return None
    except Exception as e:
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
        st.error(f"An error occurred while evaluating changes: {e}")
        return "Error"

def main():
    st.title("Website Content Change Evaluator Using Wayback Machine")
    st.write("""
        Upload a CSV file containing URLs, before dates, and after dates.
        Define which columns correspond to each field.
        The app will fetch the content from each URL on the specified dates using the Wayback Machine and evaluate the changes.
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
                before_date_col = st.selectbox("Select the Before Date column (YYYY-MM-DD)", options=df.columns)
                after_date_col = st.selectbox("Select the After Date column (YYYY-MM-DD)", options=df.columns)
                submit_mapping = st.form_submit_button("Submit")

            if submit_mapping:
                # Validate that selected columns are unique
                if len({url_col, before_date_col, after_date_col}) < 3:
                    st.error("Please select three distinct columns for URL, Before Date, and After Date.")
                else:
                    st.success("Column mapping saved!")
                    
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
                        
                        # Fetch before snapshot
                        before_snapshot_url = get_wayback_snapshot(url, before_date)
                        if before_snapshot_url:
                            before_content = fetch_content_from_snapshot(before_snapshot_url)
                            if not before_content:
                                before_error = "Failed to extract content."
                        else:
                            before_content = None
                            before_error = "No snapshot found."
                        
                        # Fetch after snapshot
                        after_snapshot_url = get_wayback_snapshot(url, after_date)
                        if after_snapshot_url:
                            after_content = fetch_content_from_snapshot(after_snapshot_url)
                            if not after_content:
                                after_error = "Failed to extract content."
                        else:
                            after_content = None
                            after_error = "No snapshot found."
                        
                        # Determine evaluation status
                        if not before_content or not after_content:
                            if not before_snapshot_url and not after_snapshot_url:
                                evaluation = "No Snapshots Found"
                            elif not before_snapshot_url:
                                evaluation = "No 'Before' Snapshot"
                            elif not after_snapshot_url:
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
