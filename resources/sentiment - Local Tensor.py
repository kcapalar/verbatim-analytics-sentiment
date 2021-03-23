import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import altair as alt
import base64
import sent_clean
import spacy
from spacy.matcher import PhraseMatcher
import SessionState
import topwords_sent
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#Settings
alt.renderers.set_embed_options(padding={"left": 0, "right": 0, "bottom": 0, "top": 0})

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def read_file(filename_with_ext: str, sheet_name=0):
    ext = filename_with_ext.partition('.')[-1]
    if ext == 'xlsx':
        file = pd.read_excel(filename_with_ext, sheet_name=sheet_name)
        return file
    elif ext == 'csv':
        file = pd.read_csv(filename_with_ext, low_memory=False)
        return file
    elif ext == 'sas7bdat':
        file = pd.read_sas(filename_with_ext, format='sas7bdat', encoding='ISO-8859-1')
        return file
    else:
        print('Cannot read file: Convert to .xlsx, .csv or .sas7bdat')

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def preprocess(df, content_col):
    df = df[df['VALID_INVALID'].str.upper()=='VALID']
    df = sent_clean.init_clean(df, col=content_col)
    #Converts to Sentence and fill with 'blank' those real blank cells
    df['sentence'] = df['init_cleaned'].map({'..':'blank', '...':'blank'}).fillna(df[content_col]).str.split('.')
  	#replace comments with elipsis or .. with BLANK
    df['sentence'] = df['sentence'].fillna('blank')
    new_df = df.explode('sentence').dropna(subset=['sentence']).query("sentence!=''").reset_index()
    sent_df = new_df[['sentence', 'index']]

	#Cleans text
    cleaned_df = sent_clean.final_clean(sent_df, col='sentence')
    return cleaned_df, new_df

def nn_model(filtered):
    sent_test = filtered

    #Preprocessing Data
    max_features = 50000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(sent_test))
    list_tokenized_test = tokenizer.texts_to_sequences(sent_test)

    maxlen = 100
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    #Predicting using saved model
    lstm = tf.keras.models.load_model(r"lstm.h5")
    cnn = tf.keras.models.load_model(r"cnn_1.h5")
    cnn_gru = tf.keras.models.load_model(r"cnn_gru_1.h5")
    bi_gru = tf.keras.models.load_model(r"bi_gru_1.h5")


    #Combining predictions from four neural networks
    proba = [
        lstm.predict([X_te], batch_size=1024, verbose=1),
        cnn.predict([X_te], batch_size=1024, verbose=1),
        cnn_gru.predict([X_te], batch_size=1024, verbose=1),
        bi_gru.predict([X_te], batch_size=1024, verbose=1)
            ]

    results = pd.DataFrame(np.mean(proba, axis=0), index=sent_test.index).apply(np.argmax, axis=1).map({0:'Negative', 1:'Neutral', 2:'Positive'})
    return results

def run_model(cleaned_df):
    	filtered = cleaned_df[['cleaned']].query("(cleaned!='')&(cleaned!='blank')")['cleaned']
	df = cleaned_df[['sentence', 'cleaned','index']]\
    	.merge(nn_model(filtered).reset_index()[[0]]\
    	.rename(columns={0:'pred'}),
    			right_index=True,
    			left_index=True,
    			how='left')
	return df

def sent_tally(out, cust_col):
	#total sentiment count
    src = pd.DataFrame(out['pred'].value_counts()).reset_index().rename(columns={'pred':'count', 'index':'sentiment'})
    row_sent = alt.Chart(src).mark_bar().encode(x=alt.X('sentiment',
                                                     type='nominal',
                                                     sort=None,
                                                     axis=alt.Axis(labelAngle=360,
                                                                   title="")),
                                            	y=alt.Y('count',
                                                     type='quantitative',
                                                     axis=alt.Axis(title="Count")),
                                             color=alt.Color('sentiment:N',
                                                             scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],
                                                                            range=['firebrick','gray', 'darkgreen'])),
                                             tooltip=['sentiment:N', 'count:Q'])
	#venn diagram of sentiment - customer level
    venn = pd.DataFrame(out.fillna('None').groupby([cust_col,'pred'])['pred'].nunique()).unstack('pred').reset_index().fillna(0)
    venn.columns = [venn.columns.values[0][0],venn.columns.values[1][1],venn.columns.values[2][1],venn.columns.values[3][1],venn.columns.values[4][1]]
    venn['sum'] = venn[['Negative', 'Neutral', 'Positive']].sum(axis=1)
    venn['sentiment'] = np.where((venn['sum']==1)&(venn['Negative']==1),'Negative', 'No Comment')
    venn['sentiment'] = np.where((venn['sum']==1)&(venn['Positive']==1),'Positive', venn['sentiment'])
    venn['sentiment'] = np.where((venn['sum']==1)&(venn['Neutral']==1),'Neutral', venn['sentiment'])
    venn['sentiment'] = np.where((venn['sum']>1), 'Mixed', venn['sentiment'])
    venn1 = pd.DataFrame(venn['sentiment'].value_counts()).reset_index().rename(columns={'sentiment':'count', 'index':'sentiment'}).set_index('sentiment').reindex(['Negative','Mixed','Positive','Neutral','No Comment']).reset_index().fillna(0)
    venn1['Percent of Total'] = venn1['count'].divide(venn1['count'].sum())
    cust_sent = alt.Chart(venn1).mark_bar().encode(x=alt.X('sentiment',
                                                        type='nominal',
                                                        sort=None,
                                                        axis=alt.Axis(labelAngle=360,
                                                                      title="")),
                                                y=alt.Y('Percent of Total',
                                                        type='quantitative',
                                                        axis=alt.Axis(format='%')),
                                                color=alt.Color('sentiment:N',
                                                                scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],
                                                                                range=['firebrick','gray', 'darkgreen'])),
                                                tooltip=['sentiment:N', 'Percent of Total:Q', 'count:Q'])
    fig = cust_sent.properties(height=300, width=300, title='Sentiment by Customer') | row_sent.properties(height=300, width=300, title='All Sentiment')
    net_sent_score = (int(src.query("sentiment=='Positive'")['count']) - int(src.query("sentiment=='Negative'")['count'])) / int(src.query("sentiment!='Neutral'")['count'].sum())
    st.markdown(f"### Net Sentiment Score: {(net_sent_score*100):,.2f}%")
    st.markdown(" ")
    sent1, sent2 = st.beta_columns(2)
    with sent1:
        fig = cust_sent.properties(height=300, width=300, title='Sentiment by Customer')
        st.altair_chart(fig, use_container_width=True)
        st.markdown(get_table_download_link(venn1, "Sentiment by Customer table"), unsafe_allow_html=True)
    with sent2:
        fig = row_sent.properties(height=300, width=300, title='All Sentiment')
        st.altair_chart(fig, use_container_width=True)
        st.markdown(get_table_download_link(src, "All Sentiment table"), unsafe_allow_html=True)

def relatedwords(out):
	out['category_id'] = out['pred'].fillna('blank').factorize()[0]
	category_id_df = out[['pred', 'category_id']].fillna('blank').drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_df.values)
	out['cleaned'] = out['cleaned'].str.replace('blank', '',regex=True, case=False)
	tfidf = TfidfVectorizer(sublinear_tf=True,
	                        min_df=5,
	                        norm='l2',
	                        ngram_range=(1, 3),
	                        stop_words='english')
	features = tfidf.fit_transform(out.cleaned).toarray()
	labels = out.category_id
	corr_dict1 = {}

	for Sentiment, category_id in sorted(category_to_id.items()):
	    if Sentiment != 'blank':
	        features_chi2 = chi2(features, labels == category_id)
	        indices = np.argsort(features_chi2[0])[::-1]
	        feature_names = np.array(tfidf.get_feature_names())[indices]
	        #N-grams
	        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
	        uni_score = [score for v,score in zip(feature_names, features_chi2[0][indices]) if len(v.split(' ')) == 1]
	        uni_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices]) if len(v.split(' ')) == 1]
	        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
	        bi_score = [score for v,score in zip(feature_names, features_chi2[0][indices]) if len(v.split(' ')) == 2]
	        bi_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices]) if len(v.split(' ')) == 2]
	        trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
	        tri_score = [score for v,score in zip(feature_names, features_chi2[0][indices]) if len(v.split(' ')) == 3]
	        tri_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices]) if len(v.split(' ')) == 3]
	        corr_dict1.update({f"{Sentiment}_uni":unigrams,f"{Sentiment}_uni_score":uni_score,f"{Sentiment}_uni_pval":uni_pval,
	                          f"{Sentiment}_bi":bigrams,f"{Sentiment}_bi_score":bi_score,f"{Sentiment}_bi_pval":bi_pval,
	                          f"{Sentiment}_tri":trigrams, f"{Sentiment}_tri_score":tri_score,f"{Sentiment}_tri_pval":tri_pval})
	
	#Count of Words
	count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
	count_data = count_vectorizer.fit_transform(out.cleaned)
	vector = pd.DataFrame(zip(count_vectorizer.get_feature_names(), count_data.sum(0).getA1()))
	vector.columns = ['word','count']

	#Plot
	ngrams = ['uni', 'bi', 'tri']
	sents = ['Positive', 'Negative', 'Neutral']
	for sent in sents:
		li = []
		for ngram,name in zip(ngrams, ['Unigram', 'Bigram', 'Trigram']):
			corr = pd.DataFrame({k:v for k,v in corr_dict1.items() if f'{sent}_{ngram}' in k})
			df_ngram = corr.head(30).merge(vector, left_on=f'{sent}_{ngram}' , right_on='word', how='left')
			# df_ngram = corr[corr[f"{sent}_{ngram}_pval"]<0.05].head(30).merge(vector, left_on=f'{sent}_{ngram}' , right_on='word', how='left')
			bars = alt.Chart(df_ngram).mark_bar().encode(x="count:Q",y=alt.Y('word', type='nominal', sort=None, axis=alt.Axis(title="")),
														 tooltip=['word:N', 'count:Q'],
														  color=alt.condition(alt.datum[f'{sent}_{ngram}_pval']<0.001, alt.value('firebrick'), alt.value('grey')))
			text = bars.mark_text(align='left', baseline='middle', dx=3).encode(text='count:Q')
			fig = (bars + text).properties(height=500, width=200, title=f"{sent} {name}")
			li.append(fig)
		figs = li[0]|li[1]|li[2]
		st.altair_chart(figs)

def relatedwords1(out):
	out['category_id'] = out['pred'].fillna('blank').factorize()[0]
	category_id_df = out[['pred', 'category_id']].fillna('blank').drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_df.values)
	out['cleaned'] = out['cleaned'].str.replace('blank', '',regex=True, case=False)
	tfidf = TfidfVectorizer(sublinear_tf=True,
	                        min_df=5,
	                        norm='l2',
	                        ngram_range=(1, 3),
	                        stop_words='english')
	features = tfidf.fit_transform(out.cleaned).toarray()
	labels = out.category_id
	corr_dict = {}
	ngram_list = [i for i in tfidf.get_feature_names()]
	for Sentiment, category_id in sorted(category_to_id.items()):
	    if Sentiment != 'blank':
	        features_chi2 = chi2(features, labels == category_id)
	        indices = np.argsort(features_chi2[0])[::-1]
	        feature_names = np.array(tfidf.get_feature_names())[indices]
	        #N-grams
	        unigrams = [v for v in feature_names]
	        uni_score = [score for v,score in zip(feature_names, features_chi2[0][indices])]
	        uni_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices])]
	        corr_dict.update({f"{Sentiment}_sent":unigrams,f"{Sentiment}_score":uni_score,f"{Sentiment}_pval":uni_pval})
            
	#Count of Words
	count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
	count_data = count_vectorizer.fit_transform(out.cleaned)
	vector = pd.DataFrame(zip(count_vectorizer.get_feature_names(), count_data.sum(0).getA1()))
	vector.columns = ['word','count']
	#Plot
	sents = ['Positive', 'Negative', 'Neutral']
	li = []
	for sent in sents:
		corr = pd.DataFrame({k:v for k,v in corr_dict.items() if sent in k})
		df_ngram = corr.head(30).merge(vector, left_on=f'{sent}_sent' , right_on='word', how='left')
		# df_ngram = corr[corr[f"{sent}_pval"]<0.05].head(30).merge(vector, left_on=f'{sent}_sent' , right_on='word', how='left')
		bars = alt.Chart(df_ngram).mark_bar().encode(x="count:Q",y=alt.Y('word', type='nominal', sort=None, axis=alt.Axis(title="")),
													 tooltip=['word:N', 'count:Q'],
													  color=alt.condition(alt.datum[f'{sent}_pval']<0.001, alt.value('firebrick'), alt.value('grey')))
		text = bars.mark_text(align='left', baseline='middle', dx=3).encode(text='count:Q')
		fig = (bars + text).properties(height=500, width=200, title=f"{sent}")
		li.append(fig)
	figs = li[0]|li[1]|li[2]
	# col1, col2 = st.beta_columns([1,0])
	# with col1:
	st.altair_chart(figs, use_container_width=True)
	return ngram_list

def get_table_download_link(df, name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv" style="font-size: 10px">Download {name}</a>'
    return href

def word_viewer(data, words, content_col, cust_col):
    total_pop = data[cust_col].nunique()
    df_viewed = data[data["sentence"].str.contains(words, na=False, case=False, regex=False)][[cust_col ,content_col, 'pred']].drop_duplicates().reset_index(drop=True)
    curr_pop = df_viewed[cust_col].nunique()
    st.markdown(get_table_download_link(df_viewed, "search results"), unsafe_allow_html=True)
    st.markdown(f"There are {curr_pop:,}/{total_pop:,} respondents found.")
    return st.table(df_viewed)

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def get_channels(df, channels):
	nlp = spacy.load("en_core_web_sm")
    #Patter Matcher
	def on_match(matcher, doc, id, matches):
		return [nlp.vocab.strings[match_id] for match_id,start, end in matches]
	matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
	for k,v in channels.items():
	    for item in [word.strip() for sentence in v for word in sentence.split(',')]:
	        matcher.add(str(k), on_match, nlp(str(item)))
	
	df['channel'] = df['sentence'].str.lower().map(lambda text: [nlp.vocab.strings[match_id[0]] for match_id in matcher(nlp(text))])
	for i in [j for j in channels.keys()]:
		df[i] = df['channel'].map(lambda x: 1 if str(i) in x else 0)
	return df

def plot_channels(df,channels,cust_col):
	channel_list = [j for j in channels.keys()]
	a = df.groupby([cust_col, 'pred'])[channel_list].sum().clip(0,1).groupby('pred')[channel_list].sum().reindex(['Neutral', 'Positive', 'Negative'])
	b = a.unstack().reset_index().rename(columns={'level_0':'Channel', 0:'count'})
	c = (a.divide(a.sum())*100).unstack().reset_index().rename(columns={'level_0':'Channel', 0:'percent'})
	source = b.merge(c, on=['Channel', 'pred'])
	st.markdown(get_table_download_link(source, "channels table"), unsafe_allow_html=True)
	bars = alt.Chart(source).mark_bar().encode(
	    x=alt.X('count:Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Channel:N', sort='-x', axis=alt.Axis(title="")),
	    tooltip=['count:Q', 'Channel:N', 'pred:N', 'percent:Q'],
	    color=alt.Color('pred',scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],range=['firebrick','gray', 'darkgreen']), legend=alt.Legend(title="Sentiment"))
	)
    
	text = alt.Chart(source).mark_text(dx=15, dy=0, color='black').encode(
	    x=alt.X('sum(count):Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Channel:N', sort='-x', axis=alt.Axis(title="")),
	    detail='Channel:N',
	    text=alt.Text('sum(count):Q', format=',.0f')
	)
	fig = (bars + text).properties(title='By Channel', height=200)
	return st.altair_chart(fig)

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def get_topics(df, topics):
	nlp = spacy.load("en_core_web_sm")
    #Patter Matcher
	def on_match(matcher, doc, id, matches):
		return [nlp.vocab.strings[match_id] for match_id,start, end in matches]
	matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
	for k,v in topics.items():
	    for item in [word.strip() for sentence in v for word in sentence.split(',')]:
	        matcher.add(str(k), on_match, nlp(str(item)))
	df['topics'] = df['sentence'].str.lower().map(lambda text: [nlp.vocab.strings[match_id[0]] for match_id in matcher(nlp(text))]).apply(lambda y: [str('Others')] if len(y)==2 else y)
	df['topics'] = np.where(df['sentence']=='blank', [str('No Feedback')], df['topics'])
	df['pred'] = np.where(df['topics']=='No Feedback', 'Neutral', df['pred'])
	for i in [j for j in topics.keys()] + ['Others', 'No Feedback']:
		df[i] = df['topics'].map(lambda x: 1 if str(i) in x else 0)
	return df

def plot_topics(df, topic_list, cust_col):
    a = df.groupby([cust_col, 'pred'])[topic_list].sum().clip(0,1).groupby('pred')[topic_list].sum().reindex(['Neutral', 'Positive', 'Negative'])
    b = a.unstack().reset_index().rename(columns={'level_0':'Topic', 0:'count'})
    c = (a.divide(a.sum())*100).unstack().reset_index().rename(columns={'level_0':'Topic', 0:'percent'})
    source = b.merge(c, on=['Topic', 'pred']).query("count>0")
    st.markdown(get_table_download_link(source, "topics table"), unsafe_allow_html=True)
    bars = alt.Chart(source).mark_bar().encode(
	    x=alt.X('count:Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Topic:N', sort='-x', axis=alt.Axis(title="")),
	    tooltip=['count:Q', 'Topic:N', 'pred:N', 'percent:Q'],
	    color=alt.Color('pred',scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],range=['firebrick','gray', 'darkgreen']),legend=alt.Legend(title="Sentiment"))
	)
    text = alt.Chart(source).mark_text(dx=15, dy=0, color='black').encode(
	    x=alt.X('sum(count):Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Topic:N', sort='-x', axis=alt.Axis(title="")),
	    detail='Topic:N',
	    text=alt.Text('sum(count):Q', format=',.0f')
	)
    fig = (bars + text).properties(title='By Topic', height=300, width=600)
    return st.altair_chart(fig, use_container_width=True)

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def get_products(df, products):
	nlp = spacy.load("en_core_web_sm")
    #Patter Matcher
	def on_match(matcher, doc, id, matches):
		return [nlp.vocab.strings[match_id] for match_id,start, end in matches]
	matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
	for k,v in products.items():
	    for item in [word.strip() for sentence in v for word in sentence.split(',')]:
	        matcher.add(str(k), on_match, nlp(str(item)))
	
	df['products'] = df['sentence'].str.lower().map(lambda text: [nlp.vocab.strings[match_id[0]] for match_id in matcher(nlp(text))])
	for i in [j for j in products.keys()]:
		df[i] = df['products'].map(lambda x: 1 if str(i) in x else 0)
	return df

def plot_products(df, products, cust_col):
    prod_list = [j for j in products.keys()]
    a = df.groupby([cust_col, 'pred'])[prod_list].sum().clip(0,1).groupby('pred')[prod_list].sum().reindex(['Neutral', 'Positive', 'Negative'])
    b = a.unstack().reset_index().rename(columns={'level_0':'Product', 0:'count'})
    c = (a.divide(a.sum())*100).unstack().reset_index().rename(columns={'level_0':'Product', 0:'percent'})
    source = b.merge(c, on=['Product', 'pred'])
    st.markdown(get_table_download_link(source, "products table"), unsafe_allow_html=True)
    bars = alt.Chart(source).mark_bar().encode(
	    x=alt.X('count:Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Product:N', sort='-x', axis=alt.Axis(title="")),
	    tooltip=['count:Q', 'Product:N', 'pred:N', 'percent:Q'],
	    color=alt.Color('pred',scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],range=['firebrick','gray', 'darkgreen']), legend=alt.Legend(title="Sentiment"))
	)
    
    text = alt.Chart(source).mark_text(dx=15, dy=0, color='black').encode(
	    x=alt.X('sum(count):Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Product:N', sort='-x', axis=alt.Axis(title="")),
	    detail='Product:N',
	    text=alt.Text('sum(count):Q', format=',.0f')
	)
    fig = (bars + text).properties(title='By Product', height=200)
    return st.altair_chart(fig)
  
def read_keywords():
    keywords = pd.read_excel(r'.\resources\Keywords.xlsx')
    channels = keywords.query("channel_tag==1")[['Topic','Keywords']].set_index('Topic').T.to_dict('list')
    topics = keywords.query("topic_tag==1")[['Topic','Keywords']].set_index('Topic').T.to_dict('list')
    products = keywords.query("product_tag==1")[['Topic','Keywords']].set_index('Topic').T.to_dict('list')
    return channels, topics, products

def get_topwords(data, content_col, columns):
    for items in columns.keys():
        try:
            cond1 = data[items]==1
            cond2 = data[content_col].str.lower().str.contains('|'.join([i for i in [items.lower()[:-1],items.lower()+'ing',items.lower()+'s',items.lower()]]))
            topw = topwords_sent.main(data[cond1&(cond2==False)], content_col)
            st.markdown(items)
            st.dataframe(topw)
        except:
            pass

def matrix(df, t, score_col):
    ave_score = df[score_col].mean()
    ave_count = df[t].sum().mean()
    summ = pd.DataFrame({'Occurence':ave_count, 'NPS Score':ave_score}, index =['Overall']) 
    for i in t:
        nps = df[df[i]==1][score_col].mean()
        count = df[df[i]==1][i].sum()
        summ.loc[i] = [count, nps]
    scatter = alt.Chart(summ.reset_index()[1:]).mark_circle(size=60).encode(x='Occurence',
                                                                         y='NPS Score',
                                                                         color=alt.Color('index', legend=alt.Legend(title="Topics")),
                                                                         tooltip=['index', 'Occurence', 'NPS Score'],
                                                                         text=alt.Text('index:N')).interactive()
    vline = alt.Chart(summ.head(1)).mark_rule(color='black',
                                              strokeWidth=1,
                                              strokeDash=[3,5]).encode(x='Occurence:Q')
    hline = alt.Chart(summ.head(1)).mark_rule(color='black',
                                              strokeWidth=1,
                                              strokeDash=[3,5]).encode(y='NPS Score:Q')
    vtext = alt.Chart(pd.DataFrame({'Occurence':[summ.iloc[0]['Occurence']], 'NPS Score':[0.5]})).mark_text(text='Average Occurence',
                                                                                                            baseline='middle',
                                                                                                            align='left',
                                                                                                            dx=5).encode(x='Occurence:Q',
                                                                                                                         y='NPS Score:Q').interactive()
    htext = alt.Chart(pd.DataFrame({'Occurence':[0.5], 'NPS Score':[summ.iloc[0]['NPS Score']]})).mark_text(text='Average NPS Score',
                                                                                                            baseline='middle',
                                                                                                            align='left',
                                                                                                            dy=40).encode(x='Occurence:Q',
                                                                                                                          y='NPS Score:Q').interactive()
    #Matrix Summary
    xmean = summ.iloc[0]['Occurence']
    ymean = summ.iloc[0]['NPS Score']
    topic_summ = summ[1:]
    q1 = topic_summ[(topic_summ['Occurence']>=xmean)&(topic_summ['NPS Score']>=ymean)].index
    q2 = topic_summ[(topic_summ['Occurence']<xmean)&(topic_summ['NPS Score']>ymean)].index
    q3 = topic_summ[(topic_summ['Occurence']<xmean)&(topic_summ['NPS Score']<ymean)].index
    q4 = topic_summ[(topic_summ['Occurence']>xmean)&(topic_summ['NPS Score']<ymean)].index
    layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))
    fig2 = go.Figure(data=[go.Table(header=dict(values=['Q1/Promoters Concerns', 'Q2/Least Concern of Promoters', 'Q3/Least Urgent Concern', 'Q4/Urgent Concerns'],
											fill_color='firebrick',
											font=dict(color='white'),
											line_color='darkslategray'),
								cells=dict(values=[q1, q2, q3, q4],
											fill=dict(color='white'),
											line_color='darkslategray'))], layout=layout)
    fig1 = (scatter + vline + hline)
    st.markdown("### Occurence Score Matrix")
    st.markdown("Topics plotted along two dimensions, NPS Score and number of mentions.")
    st.markdown(f"Average Topic Occurences (vertical dashed line): {int(summ.iloc[0]['Occurence'])}")
    st.markdown(f"Average NPS (horizontal dashed line): {summ.iloc[0]['NPS Score']:.2f}")
    col1, col2 = st.beta_columns(2)
    with col1:
        st.altair_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

def heatmap_numeric_w_dependent_variable(df, dependent_variable):
    '''
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    '''
    def interpret_corr(i):
        if i==-1:
            return "Perfect Negative / Inverse"
        elif i <=-0.70:
            return "Strong Negative / Inverse"
        elif i <=-0.50:
            return "Moderate Negative / Inverse"
        elif i==0:
            return "No relationship"
        elif i<=0.3:
            return "Weak Positive / Direct"
        elif i<=0.5:
            return "Moderate Positive / Direct"
        elif i<=0.7:
            return "Strong Positive / Direct"
        elif i==1:
            return "Perfect Positive / Direct"
        else:
            pass 
    a = df.corr()[[dependent_variable]].sort_values(by=dependent_variable)
    a.index = [i.split('Satisfaction_')[-1] for i in a.index]
    a['interpret'] = a['pred_num'].map(lambda x: f"{round(x, 4)} ({interpret_corr(x)})")
    a = a.reset_index().rename(columns={'index':'Attribute'})
    a['Attribute'] = a['Attribute'].str.replace("_", " ", regex=True)
    fig = alt.Chart(a).mark_rect().encode(y='Attribute:O', color='pred_num:Q')
    text = alt.Chart(a).mark_text().encode(y='Attribute:O', text='interpret:O')
    st.altair_chart((fig + text).properties(title='Correlation Score', width=400))

def load_sentiment_page(df):
    st.header("🌡️ Sentiment")
    st.markdown("* This feature allows you to extract the sentiment of customers in the selected text column.")
    columns = [col for col in df.columns]
    
    content_col = st.sidebar.selectbox("Select Text Column", (columns))
    cust_col = st.sidebar.selectbox("Select Customer ID Column", (columns), index=1)
    segment_col = st.sidebar.selectbox("Select Segment/category Column", (columns), index=2)
    segment_val = st.selectbox("Which segment/category would you like to view?", tuple(['View All'] + df[segment_col].dropna().unique().tolist()))
    score_col = st.sidebar.selectbox("Select Score Column", (columns), index=3)

    session_state = SessionState.get(checkboxed=False)
    if st.sidebar.button("Confirm") or session_state.checkboxed:
        session_state.checkboxed = True
        pass
    else:
        st.stop() 
    
    # st.markdown("Preprocessing DataFrame")
    cleaned_df, new_df = preprocess(df, content_col)
    pred_df = run_model(cleaned_df)
    out = pred_df.drop(columns=['sentence']).merge(new_df, how='right', on='index', copy=False).drop_duplicates(subset=[cust_col,'sentence'])

    #Run Dashboard =====
    if segment_val!='View All':
        segdf = out[out[segment_col]==segment_val]
    else:
        segdf = out.copy()  
    icon1, text1, icon2, text2 = st.beta_columns([1,10,1,10])  
    with icon1:  
    	img1 = Image.open(r'.\resources\respondents.png')
    	st.image(img1, use_container_width=True)
    with text1:
        st.markdown(f"### Respondents:  {segdf[cust_col].nunique():,}")
    with icon2:
    	img2 = Image.open(r'.\resources\comments.png')
    	st.image(img2, use_container_width=True)
    with text2:
        st.markdown(f"### Response:  {df.shape[0]:,}") #{segdf['index'].nunique():,}
    
    #Sentiment
    segdf['pred'] = np.where(segdf[content_col].isna(), np.nan, segdf['pred'])
    sent_tally(segdf, cust_col)
    
    #Related Words
    st.subheader("Related Words to Sentiment")
    st.markdown("This section shows the words related to the sentiments (in order) and their corresponding number of occurences")
    ngram_list = relatedwords1(segdf)
    with st.beta_expander("- See Ngram breakdown"):
    	relatedwords(segdf)
    
    
    channels, topics, products = read_keywords()
    ch_df = get_channels(segdf, channels)
    prod_df = get_products(ch_df, products)
    top_df = get_topics(prod_df, topics)
    
    st.subheader("Mentions")
    col1, col2 = st.beta_columns(2)
    #Channels
    with col1:
        plot_channels(ch_df, channels, cust_col)
        with st.beta_expander("- Browse topwords"):
            get_topwords(ch_df, content_col, channels)
    #Products
    with col2:
       plot_products(prod_df, products, cust_col)
       with st.beta_expander("- Browse topwords"):
           get_topwords(prod_df, content_col, products)

    #Topics
    topic_list = [j for j in topics.keys()] + ['Others', 'No Feedback']
    remove_words = st.multiselect("Search words to remove: ", options=(topic_list))
    if len(remove_words)>0:
        mask_topic = [i for i in topic_list if i not in remove_words]
        plot_topics(top_df, mask_topic, cust_col)
    else:
    	plot_topics(top_df, topic_list, cust_col)
    with st.beta_expander("- Browse topwords"):
        get_topwords(top_df, content_col, topics)

    #Browse
    with st.beta_expander("- Browse responses per category"):
        col1, col2 = st.beta_columns(2)
        ref = {"Channel":channels, "Product":products, "Topics":topics}
        with col1:
            cat = st.selectbox("Category:", tuple(ref.keys()))
        with col2:
            cat_val = st.selectbox("Subcategory:", tuple([k.strip() for v in ref[cat] for k in v.split(',')]+['Others']))
        st.table(top_df[top_df[cat_val]==1][[cust_col, content_col, 'pred', cat_val]].drop_duplicates())
    
    #Topic Grade Matrix
    matrix(top_df, [i for i in topic_list if i not in ['No Feedback']], score_col) 
    
    #Correlation of attribute columns to sentiment   
    st.subheader("Correlation of attribute columns to sentiment")
    attrib = top_df[[i for i in top_df.columns.tolist() if i.startswith('Satisfaction')] + ['pred']]
    attrib['pred_num'] = attrib['pred'].map({'Positive':1, 'Neutral':0, 'Negative':-1})
    heatmap_numeric_w_dependent_variable(attrib.drop(columns=['pred']).dropna(axis=1, how='all'), 'pred_num')
    #Download entire table
    st.markdown(get_table_download_link(top_df, "full results"), unsafe_allow_html=True)
    
    #Word Viewer
    st.subheader("Word Viewer")
    words = st.text_input("Search content with the following words: ")
    if len(words)>0:
        word_viewer(top_df, str(words).lower(), content_col, cust_col)