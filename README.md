<h2 align="center"> 🕹️ Nocode Interface for DocsArray </h2>

<h3 align="center">🧠 Powered by <ins>Jina AI</ins><sup><a  href="https://jina.ai/"></a></sup> </h3>

DocsArray is a type agnostic Python library for effortless processing and manipulation of data by [Jina AI](https://jina.ai/). It can also be thought of as a single data structure for all kinds of data.

## 💥 Why DocsArray? 

- Python is all you need to know
- One-stop solution for all your data processing needs
- Scalable data structure for all your ML needs
- Portable and easy to integrate in production

## ⚡ Nocode Interface using Streamlit
A simple nocode interface to build a search prototype on your own data in just three simple steps.

- Plug in the data 
- Generate Embeddings with a click of a button
- Use out-of-the-box search functionality to check the results

## ⌛ Bringing up the Local Instance

🍴 Clone the `docarray-nocode` repository using the following command

```bash
git clone https://github.com/Shubhamsaboo/docarray-nocode.git
```

You need to have Python 3.7+ installed, and then you can create a python virtual environment using [pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) depending on your python setup. 

After installing and activating the virtual environment (The commands will differ for windows and Linux installations which is clearly specified in the respective documentation), you can run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

After installing the required dependencies, you can bring up the main application to validate that it's working in your local environment. To bring up the application, you can run the following command:

```bash
streamlit run app.py
```

🏁 **Note** - For now, it only supports text data. Moving forward, I will be adding support to other data types as well!

## 📚 Leraning Resources 
- [Learning Portal](https://learn.jina.ai/) 
- [DocsArray Docuemtnation](https://docarray.jina.ai/) 
- [Blog](https://jina.ai/blog/2022-01-13_DocArray/) 