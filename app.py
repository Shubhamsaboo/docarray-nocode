import streamlit as st
from docarray import DocumentArray, Document
import pandas as pd

st.title("🤖 No Code Interface for DocArray")

st.image("docarray-banner.png")

st.subheader("Load the Data 🧵")
file = st.file_uploader("Choose a .csv file")

if file is not None:
    dataframe = pd.read_csv(file)
    if st.button("Show Data"):
        st.dataframe(dataframe)

    if dataframe is not None:

        meta = st.selectbox("Select the text field", dataframe.columns)
        dataframe = dataframe.loc[:, [meta]]

        def create_docarray():
            dataframe['text'] = dataframe[meta]
            docs = DocumentArray.from_dataframe(dataframe)
            return docs

        docs = create_docarray()

        if st.button("Create DocArray"):
            # Creating a DocumentArray
            st.write("DocArray created with {} documents".format(len(docs)))
            
        st.subheader("Print the documents 🖨️")
        top_k = st.slider('Number of Docs to be printed', 0, 5)  
        with st.expander("Ready to Print"):  
            selected = docs[:top_k]
            for idx, d in enumerate(selected):
                st.write("{}".format(d.text))


        st.subheader("Embed documents via Feature Hashing 🔨")

        dim = st.select_slider('Choose the embedding Dimensions', [128, 256, 512])

        docs.apply(lambda d: d.embed_feature_hashing(n_dim=dim, fields=('text')))

        if st.button("Generate Embeddings"):
            st.write("Embeddings generated with shape {}".format(docs.embeddings.shape))

        st.subheader("Search the documents 🕵️")
        
        query = st.text_input("Enter the query")
        
        q = (Document(text=query).embed_feature_hashing(n_dim=dim, fields=('text')).match(docs, limit=3, exclude_self=True, metric='cosine', use_scipy=True))
        
        if st.button("Search"):
            st.write('Top 3 matches are:')
            st.write(q.matches[:, ('text')])

        st.subheader("Visualize the embeddings 📊")
        st.info("Clicking this button will stop the application flow and takes you to a new screen.")
        if st.button("Visualize Embeddings"):
            docs.plot_embeddings()

        