# GitHub Chatbot

## Introduction

Welcome to the GitHub repository for the GitHub Chatbot! 

## Prerequisites

Before you begin, make sure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/taham655/githubChatBot.git
```
## Create a Virtual Env

```bash
python -m venv env
```
Now activate it

```bash
source env/bin/activate
```
### Install Required Dependencies

```bash
pip install -r requirements.txt
```
## Creating an environment variable
### create a '.env' and add the following:
with your api keys :p
```bash
HUGGINGFACEHUB_API_TOKEN=

REPLICATE_API_TOKEN =

TOKENIZERS_PARALLELISM=true
```



## Running Streamlit 
Now time to run the app
```bash
streamlit run app.py
```



