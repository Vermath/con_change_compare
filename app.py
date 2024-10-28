import streamlit as st
import pandas as pd
import requests
import openai
from datetime import datetime
from urllib.parse import urlparse, unquote

# Initialize OpenAI client using Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_base_url_and_slug(url):
    """
    Extracts the base URL and post slug from a full post URL.
    Example:
        Input: https://handletheheat.com/halloween-cookies/
        Output: base_url = https://handletheheat.com, slug = halloween-cookies
    """
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        path = parsed_url.path.strip('/')
        slug = path.split('/')[-1] if path else None
        return base_url, slug
    except Exception as e:
        st.error(f"Error parsing URL '{url}': {e}")
        return None, None

def get_post_by_slug(url, slug):
    """
    Fetch the post data from WordPress REST API using the slug.
    Returns the post ID if found, else None.
    """
    try:
        api_endpoint = f"{url}/wp-json/wp/v2/posts?slug={slug}"
        response = requests.get(api_endpoint)

        if response.status_code == 200:
            data = response.json()
            if data:
                post = data[0]
                post_id = post.get('id')
                return post_id
            else:
                st.warning(f"No post found with slug '{slug}' at URL: {url}")
                return None
        else:
            st.error(f"Failed to fetch post with slug '{slug}' from {url}. Status Code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching post by slug: {e}")
        return None

def check_revisions_enabled(url, post_id):
    """
    Check if revisions are enabled for the given post.
    Returns True if revisions are available, else False.
    """
    try:
        revisions_endpoint = f"{url}/wp-json/wp/v2/posts/{post_id}/revisions"
        response = requests.get(revisions_endpoint)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return True
            else:
                return False
        elif response.status_code == 404:
            # Revisions endpoint not found
            return False
        else:
            st.error(f"Failed to check revisions for post ID {post_id}. Status Code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"An error occurred while checking revisions: {e}")
        return False

def fetch_content_by_date(url, post_id, target_date):
    """
    Fetch the post content as of the specified date using revisions.
    Returns the content if found, else None.
    """
    try:
        # Fetch all revisions
        revisions_endpoint = f"{url}/wp-json/wp/v2/posts/{post_id}/revisions"
        response = requests.get(revisions_endpoint)

        if response.status_code == 200:
            revisions = response.json()
            if not revisions:
                st.warning(f"No revisions found for post ID {post_id} on {url}.")
                return None

            # Convert target_date to datetime object
            if isinstance(target_date, str):
                target_date_obj = pd.to_datetime(target_date)
            else:
                target_date_obj = target_date

            # Find the latest revision before or on the target_date
            suitable_revisions = [
                rev for rev in revisions
                if pd.to_datetime(rev['date']) <= target_date_obj
            ]

            if not suitable_revisions:
                st.warning(f"No revisions found for post ID {post_id} before {target_date_obj} on {url}.")
                return None

            # Get the most recent suitable revision
            latest_revision = max(suitable_revisions, key=lambda x: pd.to_datetime(x['date']))
            content = latest_revision.get('content', {}).get('rendered', '')
            return content
        else:
            st.error(f"Failed to fetch revisions for post ID {post_id}. Status Code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching content by date: {e}")
        return None

def fetch_wordpress_content(url, date):
    """
    Fetch content from a WordPress site for a specific date using the WordPress REST API.
    Checks if revisions are enabled before attempting to fetch.
    Returns the content if successful, else returns an appropriate error message.
    """
    try:
        base_url, slug = extract_base_url_and_slug(url)
        if not base_url or not slug:
            return None, "Invalid URL or unable to extract base URL and slug."

        post_id = get_post_by_slug(base_url, slug)
        if not post_id:
            return None, "Post not found."

        revisions_enabled = check_revisions_enabled(base_url, post_id)
        if not revisions_enabled:
            return None, "Revisions API is not enabled."

        content = fetch_content_by_date(base_url, post_id, date)
        if content:
            return content, None
        else:
            return None, "Content not found for the specified date."
    except Exception as e:
        return None, f"Error: {e}"

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
    st.title("WordPress Content Change Evaluator")
    st.write("""
        Upload a CSV file containing URLs, before dates, and after dates.
        Define which columns correspond to each field.
        The app will fetch the content from each URL on the specified dates and evaluate the changes.
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
                before_date_col = st.selectbox("Select the Before Date column", options=df.columns)
                after_date_col = st.selectbox("Select the After Date column", options=df.columns)
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
                        
                        # Fetch before content
                        before_content, before_error = fetch_wordpress_content(url, before_date)
                        
                        # Fetch after content
                        after_content, after_error = fetch_wordpress_content(url, after_date)
                        
                        if before_error or after_error:
                            # If there's an error in fetching either content, mark accordingly
                            if before_error == "Revisions API is not enabled" or after_error == "Revisions API is not enabled":
                                evaluation = "Revisions API is not enabled"
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
