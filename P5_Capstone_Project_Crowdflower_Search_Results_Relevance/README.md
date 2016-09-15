# Capstone Project
## Crowdflower Search Results Relevance

The goal of this project is to create a model that can be used to measure the relevance of search results.

### Install

This project requires **Python 2.7** with the following library installed:
- [pandas](http://pandas.pydata.org/)
- [numpy](http://www.numpy.org/)
- [nltk](http://www.nltk.org/data.html)
- [sklearn](http://scikit-learn.org/stable/install.html)

### Data

The datasets can be found in the `/data` folder:
`train.csv`
`test.csv`

They can also be downloaded from the [Kaggle webpage](https://www.kaggle.com/c/crowdflower-search-relevance/data).

Note: [Pickle files](https://docs.python.org/2/library/pickle.html) are also present in this folder, which were used as check points while iterating through the code. They are all the final version used to obtain the `submission.csv` output.

### Run

In a terminal or command window, run the following command:

```python main.py```  

This will run the `main.py` file and execute the machine learning code.
