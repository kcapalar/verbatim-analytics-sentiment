#Packages
import streamlit as st

def write_faq():
    st.markdown(
        """# 📙 FAQs
        
Q: Why is my homepage prompting an error as soon as opened? 
<br><i>This error is caused by missing data. Once file is loaded, the error should be resolved.</i>
\nQ: The page/chart is not updating after I changed my selections? 
<br><i>If there is an error or the contents are not refreshed, simply reselect or click again to until updates are shown. If error persists, reload the page.</i>
\nQ: Why are my settings are gone/back to default after reloading the page? 
<br><i>Hitting the reload button will reset all configurations. If you wish to refresh the page but retain all selected options, simply press 'R' or click the hamburger button on the upper right corner and select 'Run'. </i>
\nQ: My file is not in excel or csv format, why won't the data load? 
<br><i>You have to convert first the data into a .xlsx or .csv format</i>


## ❓ How it Works
### Top Keywords
* <b>Step 1: Upload data</b>
    \n In the sidebar, upload your file for text analysis by clicking browse or dropping the file. 
    The file should have a .xlsx or .csv filename extension, contain more than 1 row and have at least one column (two if with categories). 
* <b>Step 2: Select Column</b>
    \n After reading the file, select the column name where text is. A data preview is also shown in the page for a quick overview of the contents.
    You can also select to have your analysis done per category by selecting the column name where categories can be found. The default value is <i>None</i> which means an overall analysis will be conducted.
* <b>Step 3: Data Preprocessing and Top Word Extraction (automated)</b>
    \n Data cleaning process will be initiated which includes character/slang/jargon subtitutions, words removal and text normalization.
    Extracts words using ngrams (unigram, bigram, trigram) and counts its occurence in the document.
    Unigrams are one word text, bigrams are two-word text which appeared consecutively and trigrams are three-word text. 
* <b>Step 3: Configure Chart and Display Options</b>
    - Color Palette - change color palette of the bar chart from the options listed
    - Number of Keywords - modify the number of top words displayed based on occurences/count. In case of a tie, both words will be displayed and treated as one.
    - Omit Words - select which word/s you wish to remove in the chart
* <b>Step 4: Download data</b>
    \n Download chart as png image by clicking the camera button on the upper right region of the chart area. Visit this [Plotly Modebar Navigation](https://plotly.com/chart-studio-help/getting-to-know-the-plotly-modebar/) to learn more about chart buttons.  
        """, unsafe_allow_html=True
        )