# Streamlit App Deployment 

This repo is dedicated for the data product deployment via streamlit. The app enables users to predict local flight tickets fare by entering their travel details including origin airport, destination airport, departure date, departure time, and cabin type. 


Project Organization
------------

    ├── __pycache__
    ├── app.py           <- Script to run the streamlit app
    ├── README.md          <- The top-level README for developers using this project.
    ├── data                <- data to save airport coordinates and distance of all airports
    │      ├──dinstancematrices <- data for the distance for pairs of airports
    │  
    ├── mlp_utils.py       <- Scripts containing all the utilities required to run MLP models
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │  
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`


--------


# 1. Web Access

A fully functional app can be accessed online via https://amla-at3.streamlit.app.


![Demo](https://github.com/neloeblah/amla-streamlit/assets/62013607/22c80c9b-789e-4584-a27f-bf7d5230e44d)



# 2. Using the app

On the starting page of the app you will find a collapsible side bar section where you are expected to put in the travel details required for fares prediction including:

Origin Airport: ConsistS of all available local airports.

Destination Airport: Consists of all available local airports. Cannot be the same as origin airport

Departure Date: Specific format required YYYY-MM-DD

Departure Time: In 24 hours format with 30 minutes interval

Cabin Type: Consists of all available cabin type


After all the required travel details are inputted, then click on the red 'predict' button to run the 4 trained machine learning models to get the result of models prediction.

A supplementary map which lines out the origin and arrival airports are also provided at the bottom part of the apps. The line of which the origin and destination airport is selected will be highlighted in red.



# 3. Local Access

## Installation Guide

A local version can also be installed with Python 3.10 or later installed. You can install Poetry to manage package dependancies using the following command:

> pip install poetry

After installing Poetry, navigate to the project directory and run:

> poetry install

This should prepare the virtual environment and install all the necessary dependencies.



## Initiating the app locally

Activate the virtual environment with the following command:

> poetry shell

A virtual poetry shell will be built. Then move the working directory to streamlit.

To run the app use the following command:

> streamlit run app.py





