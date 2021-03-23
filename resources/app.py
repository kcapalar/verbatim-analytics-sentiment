#Packages
import streamlit as st
import pandas as pd
import topwords
import FAQs
import sentiment

st.set_page_config(
    page_title="Verbatim Analytics",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="auto")
    
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(suppress_st_warning=True, persist=True, allow_output_mutation=True)
def load_data(path_to_file):
    try:
        data = pd.read_excel(path_to_file, sheet_name=0)
    except:
        data = pd.read_csv(path_to_file, low_memory=False)
    return data
    # if path_to_file.partition('.')[-1]=='xlsx':
    #     data = pd.read_excel(path_to_file, sheet_name=0)
    # elif path_to_file.partition('.')[-1]=='csv':
    #     data = pd.read_csv(path_to_file, low_memory=False)
    # else:
    #     st.text("Use .csv or .xlsx file")
    # return data

#PAGES --------------------------------------------------------------------------------------------------------------------
def load_homepage():
    try:
        st.markdown("# 💬 Verbatim Analytics")
        st.write("This takes verbatim feedback from respondents and provides quantifiable pictures and data insights. "
                    "There are three features available found in the left tab: Topic Keywords, Topics Explorer and Customer Sentiments.")
        for i in range(2):
            st.write("")
            path_to_file = st.sidebar.file_uploader('Upload your file:', type = ['xlsx', 'csv'])
            if path_to_file:
                data = load_data(path_to_file)
                return data
    except:
        st.warning("Please select your file for text analysis")

#MAIN -----------------------------------------------------------------------------------------------
def navigate_pages():
    try:
        tabs = ["Top Keywords", "Topics Explorer", "Customer Sentiments", "FAQs"]
        page = st.sidebar.radio("Features", tabs)
        if page == tabs[0]:
            data = load_homepage()
            st.write("Data Preview")
            st.dataframe(data)
            topwords.load_topwords_page(data)
            
        elif page == tabs[1]:
            data = load_homepage()
            # sentiment.load_sentiment_page(data)
            st.markdown("# 🚧 Page under Construction")
            #Insert function for topics here
            
        elif page == tabs[2]:
            data = load_homepage()
            sentiment.load_sentiment_page(data)
            #insert function for sentiments here
            # st.markdown("# 🚧 Page under Construction")
            
        elif page == tabs[3]:
            FAQs.write_faq()
    except:
        st.warning("Confirm your selection using the by clicking the button to continue.")


navigate_pages()
