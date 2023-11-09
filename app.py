import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import torch
import datetime

from joblib import load
from mlp_utils import TorchImagePreprocessor, MLPRegressor

# Globals
AIRPORTS = ["ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAD", 
            "JFK", "LAX", "LGA", "MIA", "OAK", "ORD", "PHL", "SFO",]
CABINS = ["coach", "premium coach", "business", "first"]
MODEL_PATHS = {
    'baseline': './models/baseline-avg.csv',
    'xgboost': './models/xgb-bst2.joblib',
    'airport': './models/airport.model',
    'cabin': './models/cabin.model',
    'xgb_s_ensemble': './models/xgb_s_data.csv',
    'xgb_d_ensemble': './models/xgb_d_data.csv',
    'xgb_n_ensemble': './models/xgb_n_data.csv',
    'random_forest':'./models/rf_model_pipeline.joblib',
    'mlp': './models/mlp-l1norm-processor-kh.joblib',
    'linear_regression':'./models/simple-regression-zb-attempt3.joblib'
}


def linear_regression():
    pipeline = load("./models/simple-regression-zb-attempt3.joblib")
    return pipeline


def initialise_states():
    """
    Initialize session state variables if they don't exist.
    """
    states = ["origin", "destination", "departure_date", "departure_time", "cabin_type"]
    for s in states:
        st.session_state[s] = st.session_state.get(s, None)

    result_states = [
        "result_model0",
        "result_model1",
        "result_model2",
        "result_model3",
        "result_model4",
    ]
    for rs in result_states:
        st.session_state[rs] = st.session_state.get(rs, -1.0)

    st.session_state["cache"] = st.session_state.get("cache", "")


def get_model(name="baseline"):
    """
    Get the specified machine learning model.

    Args:
        name (str): The name of the model.

    Returns:
        object: The trained machine learning model.
    """
    model_path = MODEL_PATHS.get(name)

    if model_path.endswith(".csv"):
        model_obj = pd.read_csv(model_path)
    else:
        model_obj = load(model_path)

    return model_obj


def baseline_prediction(df):
    """
    Show a baseline flight price prediction using average cost per airports.

    Args:
        df (DataFrame): Input data containing origin and destination.

    Returns:
        float: Predicted flight price.
    """
    origin = st.session_state["origin"]
    destination = st.session_state["destination"]

    # Account for missing flight paths from training dataset
    conditions = [
        (origin == "JFK" and destination in ["EWR", "LGA"]),
        (origin == "LGA" and destination in ["EWR", "JFK"]),
        (origin == "EWR" and destination == "JFK"),
    ]

    # Only valid flights available in training set
    if any(conditions):
        origin = "EWR"
        destination = "LGA"

    # Prediction
    pred_condition = (df["startingAirport"] == origin) & (
        df["destinationAirport"] == destination
    )
    pred = df.loc[pred_condition, "totalFare"]

    return pred.values[0]


def convert_epoch_time():
    """
    Merge user inputs for time and date to get an epoch timestamp for 
    machine learning inputs.

    Returns:
        datetime.datetime: Flight time in epoch time format.
    """
    # Turn parts into datetime format
    depart_date = pd.to_datetime(st.session_state["departure_date"])
    depart_time = pd.to_datetime("1970-01-01 " + str(st.session_state["departure_time"]))

    # Combine and get timestamp
    full_timestamp = depart_date + datetime.timedelta(hours=depart_time.hour, minutes=depart_time.minute, seconds=depart_time.second) 
    epoch_timestamp = full_timestamp.timestamp()

    return epoch_timestamp


def parse_user_inputs():
    """
    Parse and reformat user inputs for flight price prediction.

    This function takes user inputs for origin, destination, departure date, departure time, and cabin type,
    and converts them into a structured DataFrame suitable for models. It reformats the date and converts
    the time from user input into epoch format.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed user inputs.
    """
    # Reformat date
    depart_date = pd.to_datetime(st.session_state["departure_date"])

    # Convert time from User Input into epoch format for models
    epoch_timestamp = convert_epoch_time()

    # Create input dataframe
    input_dict = {
        "startingAirport": st.session_state["origin"],
        "destinationAirport": st.session_state["destination"],
        "segmentsCabinCode": st.session_state["cabin_type"],
        "isNonStop": 1,
        "segmentsDepartureTimeEpochSeconds": epoch_timestamp,
        "totalTravelDistance": np.nan,
        "year": depart_date.year,
        "month": depart_date.month,
        "day_of_month": depart_date.day,
        "day_of_week": depart_date.dayofweek,
    }
    df = pd.DataFrame.from_dict(input_dict, orient="index").T

    return df


def convert_dates(df, col):
    """
    Convert a date column in a DataFrame to multiple date-related columns.

    This function takes a DataFrame and the name of a date column as input. It then
    extracts the date components (year, month, day of the month, and day of the week)
    from the specified date column and creates new columns with these components.
    The original date column is removed from the DataFrame, and the updated DataFrame is returned.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the date column to be converted.
        col (str): The name of the date column to be processed.

    Returns:
        pd.DataFrame: The updated DataFrame with date-related columns.
    """
    # Extract date type
    label = col.replace("Date", "_")

    # Split components
    df[f"{label}month"] = df[col].dt.month
    df[f"{label}day_of_month"] = df[col].dt.day
    df[f"{label}day_of_week"] = df[col].dt.day_of_week

    # Remove old format
    df.drop(columns=col, inplace=True)

    return df


def parse_user_inputs_linear_regression():
    """
    Parse and prepare user inputs for linear regression based flight price prediction.

    This function takes user inputs for origin, destination, departure date, departure time, and cabin type,
    and prepares them for linear regression based flight price prediction. It also confucts feature engineering based on the inputs to improve the prediction. 

    Returns:
        pd.DataFrame: A DataFrame containing the prepared user inputs and additional engineered features.
    """
    depart_date = pd.to_datetime(st.session_state["departure_date"])
    search_date = pd.to_datetime("today")

    if depart_date < search_date:
        search_date = depart_date

    cabin_map = {
        'first': 4,
        'business': 3,
        'premium coach': 2,
        'coach': 1
    }

    # Load the airport distances csv
    distances_df = pd.read_csv(
        "./data/Distance_of_All_Airports_20231022_000915.csv"
    )

    # Extract 'startingAirport' and 'destinationAirport' from the user inputs
    starting_airport = st.session_state["origin"]
    destination_airport = st.session_state["destination"]

    # Get the distance from 'distances_df' based on 'ORIGIN' and 'DEST' columns
    distance_travelled = distances_df.loc[
        (distances_df["ORIGIN"] == starting_airport)
        & (distances_df["DEST"] == destination_airport)
    ]["DISTANCE IN MILES"].values[0]

    input_dict = {
        "segmentsCabinCode": cabin_map[st.session_state["cabin_type"]],
        "day_diff": (depart_date - search_date).days,
        "search_day_of_week": search_date.dayofweek,
        "flight_day_of_week": depart_date.dayofweek,
        "num_segments": 1,
        "distanceTravelled": distance_travelled,
    }

    df = pd.DataFrame.from_dict(input_dict, orient="index").T

    return df

def parse_user_inputs_xgb():
    """
    Parse and prepare user inputs for XGBoost-based flight price prediction.

    This function takes user inputs for origin, destination, departure date, departure time, and cabin type,
    and prepares them for XGBoost-based flight price prediction. It also retrieves additional components
    for XGBoost predictions from external models.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared user inputs and additional components for XGBoost-based models.
    """
    # Ensure data type is correct
    depart_date = pd.to_datetime(st.session_state["departure_date"])
    search_date = pd.to_datetime("today")

    # Assignment tests allow backdated predictions. Assume search date is historical as well.
    if depart_date < search_date:
        search_date = depart_date

    # Change User input time to epoch format to match training data
    epoch_timestamp = convert_epoch_time()

    # Merge formatted User Inputs to dataframe
    input_dict = {
        "searchDate": search_date,
        "flightDate": depart_date,
        "startingAirport": st.session_state["origin"],
        "destinationAirport": st.session_state["destination"],
        "segmentsDepartureTimeEpochSeconds": epoch_timestamp,
        "segmentsCabinCode": st.session_state["cabin_type"],
    }

    df = pd.DataFrame.from_dict(input_dict, orient="index").T
    df["segmentsDepartureTimeEpochSeconds"] = pd.to_numeric(
        df["segmentsDepartureTimeEpochSeconds"]
    )

    # Adjust date formatting
    for col in ["searchDate", "flightDate"]:
        df = convert_dates(df, col)

    # Additional components for XGBoost predictions
    xgb_n_model = get_model("xgb_n_ensemble")
    xgb_d_model = get_model("xgb_d_ensemble")
    xgb_s_model = get_model("xgb_s_ensemble")

    # Combine with User Inputs
    df = df.merge(xgb_d_model, how="left", on=["startingAirport", "destinationAirport"])
    df = df.merge(
        xgb_s_model,
        how="left",
        on=["startingAirport", "destinationAirport", "flight_day_of_week"],
    )
    df = df.merge(xgb_n_model, how="left", on=["startingAirport", "destinationAirport"])

    # Convert strings to Vector Embeddings
    airport_model = get_model("airport")
    cabin_model = get_model("cabin")
    for col in ["startingAirport", "destinationAirport"]:
        df[col] = airport_model.wv[df[col]]

    df["segmentsCabinCode"] = cabin_model.wv[df["segmentsCabinCode"]]

    return df


def parse_user_inputs_rf():
    """
    Parse and prepare user inputs for Random Forest.

    This function takes user inputs for origin, destination, departure date, departure time, and cabin type,
    and prepares them for random forest prediction. It also process the input for specific data engineering required.

    Returns:

    """
    # Ensure data type is correct
    depart_date = pd.to_datetime(st.session_state["departure_date"])
    search_date = pd.to_datetime("today")

    # Assignment tests allow backdated predictions. Assume search date is historical as well.
    if depart_date < search_date:
        search_date = depart_date

    day_diff = (depart_date - search_date).days

    cabin_map = {"first": 4, "business": 3, "premium coach": 2, "coach": 1}

    def get_time_period(time):
        if 0 <= time < 6:
            departure_period = "midnight"
        elif 6 <= time < 12:
            departure_period = "morning"
        elif 12 <= time < 18:
            departure_period = "afternoon"
        elif 18 <= time < 22:
            departure_period = "evening"
        else:
            departure_period = "night"

        return departure_period

    departure_period = get_time_period(st.session_state["departure_time"].hour)

    # Merge formatted User Inputs to dataframe
    input_dict = {
        "searchDate": search_date,
        "flightDate": depart_date,
        "startingAirport": st.session_state["origin"],
        "destinationAirport": st.session_state["destination"],
        "flight_month": depart_date.month,
        "departure_period": departure_period,
        "day_diff": day_diff,
        "segmentsCabinCode": cabin_map[st.session_state["cabin_type"]],
    }

    df = pd.DataFrame.from_dict(input_dict, orient="index").T

    return df


def parse_user_inputs_mlp():
    """
    Parse and prepare user inputs for mlp model.

    This function takes user inputs for origin, destination, departure date, departure time, and cabin type, distance(min,max,mean,std),flightDate_is_holiday,flightDate_is_weekend
    It also process the input for specific data engineering required.

    Returns: df (DataDrame)

    """
    # Ensure data type is correct
    depart_date = pd.to_datetime(st.session_state["departure_date"])
    depart_df = pd.Series(depart_date)
    search_date = pd.to_datetime("today")

    # Assignment tests allow backdated predictions. Assume search date is historical as well.
    if depart_date < search_date:
        search_date = depart_date

    # Change User input time to epoch format to match training data
    epoch_timestamp = convert_epoch_time()

    cabin_map = {"first": 4, "business": 3, "premium coach": 2, "coach": 1}

    us_holidays = ["2022-04-17", "2022-05-30", "2022-07-04"]

    distance_matrix = pd.read_csv(
        f'./data/distancematrices/{st.session_state["origin"]}.csv'
    )

    input_dict = {
        "startingAirport": st.session_state["origin"],
        "destinationAirport": st.session_state["destination"],
        "isNonStop": 1,
        "segmentsDepartureTimeEpochSeconds": epoch_timestamp,
        "segmentsCabinCode": cabin_map[st.session_state["cabin_type"]],
        "day_diff": np.nan,
        "search_year": search_date.year,
        "search_month": search_date.month,
        "search_day_of_month": search_date.day,
        "search_day_of_week": search_date.dayofweek,
        "flight_year": depart_date.year,
        "flight_month": depart_date.month,
        "flight_day_of_month": depart_date.day,
        "flight_day_of_week": depart_date.dayofweek,
        "flight_weekend": (1 if depart_date.dayofweek >= 5 else 0),
        "flight_isHoliday": depart_df.dt.strftime("%Y-%m-%d")
        .isin(us_holidays)
        .astype(int),
        "distance_min": distance_matrix[
            distance_matrix["destinationAirport"] == st.session_state["destination"]
        ]["distance_min"].values[0],
        "distance_max": distance_matrix[
            distance_matrix["destinationAirport"] == st.session_state["destination"]
        ]["distance_max"].values[0],
        "distance_std": distance_matrix[
            distance_matrix["destinationAirport"] == st.session_state["destination"]
        ]["distance_std"].values[0],
        "distance_mean": distance_matrix[
            distance_matrix["destinationAirport"] == st.session_state["destination"]
        ]["distance_mean"].values[0],
    }

    df = pd.DataFrame.from_dict(input_dict, orient="index").T
    df["segmentsDepartureTimeEpochSeconds"] = pd.to_numeric(
        df["segmentsDepartureTimeEpochSeconds"]
    )

    return df


def get_remaining_airports(origin):
    """
    Create list of destination airport options to eliminate instances of users
    chosing the same airport for origin and destination, or flight paths that
    do not exist such as JFK-LGA which are both NY based. Flights from EWR to
    JFK/LGA considered valid, although only one flight exists in the training 
    data. 

    Parameters:
        origin (str): Origin airport

    Returns:
        list: Valid destination airports dependant on origin.
    """
    exclusions = [origin]
    if origin == 'JFK':
        exclusions.append('LGA')
    elif origin == 'LGA':
        exclusions.append('JFK')
    
    remaining_airports = [
            airport for airport in AIRPORTS if airport not in exclusions
        ]
    
    return remaining_airports
    

def sidebar_setup():
    """
    Set up the user interface choices in the Streamlit sidebar.

    Parameters:
        options (dict): A dictionary containing menu options for user selection, including airports, dates, and cabin types.

    Returns:
        None
    """

    with st.sidebar:
        # UI choices
        st.session_state["origin"] = st.selectbox(
            "Select Origin Airport:",
            options=AIRPORTS,
            index=0,
        )

        remaining_airports = get_remaining_airports(st.session_state["origin"])

        st.session_state["destination"] = st.selectbox(
            "Select Destination Airport:",
            options=remaining_airports,
            index=0,
        )

        st.session_state["departure_date"] = st.date_input(
            "Select Departure Date:",
            value="today",
            format="YYYY/MM/DD",
        )

        st.session_state["departure_time"] = st.time_input(
            "Select Departure Time:",
            value="now",
        )

        st.session_state["cabin_type"] = st.selectbox(
            "Select Cabin Type:",
            options=CABINS,
            index=0,
        )


def plot_flight_paths():
    """
    Generate a plot of flight paths and airport locations.

    This function generates a plot that displays flight paths and airport locations.
    It extracts flight path data from a list of airports, and adds traces for flight paths and airport locations.
    It also highlights the flight path selected based on user inputs.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure representing the flight paths and airport locations.
    """

    # Airport Co-ordinates
    df = pd.read_csv("./data/airport_coords.csv")
    df["text"] = (
        df["code"] + "<br>" + df["airport"].str.replace("Interational Airport", "")
    )

    # Initialise plot
    fig = go.Figure()

    # Extract flight paths
    lons = []
    lats = []
    n = df.shape[0] - 1
    for i in range(n):
        for j in range(df.loc[i + 1 :].shape[0]):
            lons.append(df.loc[i, "longitude"])
            lons.append(df.loc[j, "longitude"])
            lons.append(None)

            lats.append(df.loc[i, "latitude"])
            lats.append(df.loc[j, "latitude"])
            lats.append(None)

    # Plot flight paths
    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=0.5, color="blue"),
            opacity=0.3,
            name="",
            text="",
        )
    )

    # Plot aiport locations and labels
    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lon=df["longitude"],
            lat=df["latitude"],
            hoverinfo="text",
            text=df["text"],
            mode="markers",
            marker=dict(
                size=9,
                color="rgb(0, 0, 0)",
                line=dict(width=3, color="rgba(68, 68, 68, 0)"),
            ),
        )
    )

    # Highlight flight path selected from User Inputs
    selected = [st.session_state["origin"], st.session_state["destination"]]
    predicted = df.loc[df["code"].isin(selected)]
    pred_lats = [x for x in predicted["latitude"]]
    pred_lons = [x for x in predicted["longitude"]]

    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lon=pred_lons,
            lat=pred_lats,
            mode="markers+lines",
            line=dict(width=3, color="red"),
            marker=dict(
                size=10, color="red", line=dict(width=3, color="rgba(68, 68, 68, 0)")
            ),
            opacity=1,
            name="Predicted",
            text="",
        )
    )

    # Formatting
    fig.update_layout(
        title_text="Flight Paths<br>(Hover for airport names)",
        showlegend=False,
        geo=go.layout.Geo(
            scope="north america",
            projection_type="azimuthal equal area",  
            projection_scale=1.2,
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
            center=dict(lon=-100, lat=35),
        ),
        height=450,
        margin=dict(l=20, r=20, t=20, b=0, pad=0),
    )

    return fig


def update_cache_msg():
    """
    Update the cache message with user-selected flight details.

    This function constructs and updates the cache message based on user-inputs
    that were used for the last prediction. The formatted message is stored in the session state.
    """
    st.session_state[
        "cache"
    ] = """*Showing prices for **`{}`** to **`{}`**,\
 departure @ **`{}`** on **`{}`** in **`{}`** class.*""".format(
        st.session_state["origin"],
        st.session_state["destination"],
        st.session_state["departure_time"],
        st.session_state["departure_date"],
        st.session_state["cabin_type"],
    )

    return None


def display_model_results():
    """
    Summarises all predicted values into dataframe for display to user.

    Returns:
        pd.DataFrame: A dataframe containing model names and prediction values.
    """
    # Define labels and states
    names = [
        "Baseline: Average Price",
        "Model A: Linear Regression",
        "Model B: Multi-Layer Perceptron",
        "Model C: XGBoost",
        "Model D: Random Forest",
    ]
    result_states = [
        "result_model0",
        "result_model1",
        "result_model2",
        "result_model3",
        "result_model4",
    ]

    # Merge to dataframe
    display_df = pd.DataFrame(names, columns=["Models"])
    display_df["Price"] = [st.session_state[result] for result in result_states]

    # Remove invalid predictions (included for testing purposes)
    display_df.loc[display_df["Price"] < 0, "Price"] = np.nan

    return display_df


def main():
    # Streamlit title
    st.title(":airplane: Assignment 3: Data Product with Machine Learning")

    # Setup session states to manage model inputs
    initialise_states()

    # Instructions
    st.divider()
    st.markdown(
        """Welcome to the flight price prediction platform. Enter expected flight\
 details on the left side menu and hit the Predict button below to view each models'\
 expected ticket price.

Departure times are in `24hr` format and departure dates are `YYYY-MM-DD`.  

*Disclaimer: Models have analysed data between April-July. Quality outside of these\
 months may vary.* """
    )

    st.divider()

    # Set up sidebar
    sidebar_setup()

    # Train and retrieve model
    st.subheader("Results")
    col1, col2 = st.columns([0.15, 0.85])
    predict_button = col1.button("Predict", type="primary")

    if predict_button:
        st.session_state["Predict"] = True

        # Indicate to user what results are displayed
        update_cache_msg()

        # Baseline
        baseline = get_model("baseline")
        base_pred = baseline_prediction(baseline)
        st.session_state["result_model0"] = base_pred

        # Linear regression model 
        slr_model = linear_regression() # call simple linear regression function
        predict_df_slr = parse_user_inputs_linear_regression()
        st.session_state["result_model1"] = slr_model.predict(predict_df_slr)[0]

        # MLP model
        mlp_processor = get_model("mlp")
        mlp_regressor = torch.load(
            "./models/mlp-l1norm-regressor-kh.pth",
            map_location=torch.device("cpu"),
        )
        mlp_prediction = parse_user_inputs_mlp()
        mlp_prediction = mlp_processor.transform(mlp_prediction)
        st.session_state["result_model2"] = mlp_regressor.predict(mlp_prediction)[0][0]

        # XGBoost model
        xgb_model = get_model("xgboost")
        predict_df_xgb = parse_user_inputs_xgb()
        predict_df_xgb = xgb.DMatrix(predict_df_xgb)
        st.session_state["result_model3"] = xgb_model.predict(predict_df_xgb)[0]

        # Random Forest Model
        rf_model = get_model("random_forest")
        rf_prediction = rf_model.predict(parse_user_inputs_rf())
        st.session_state["result_model4"] = rf_prediction[0]

    # Display prediction description
    st.write("")
    if st.session_state["cache"] == "":
        col2.write("*No predictions have been run yet.*")
    else:
        col2.markdown(st.session_state["cache"])

    # Final Results if predictions have been run
    st.write("")
    if st.session_state["cache"] != "":
        # Summarise model predictions
        display_df = display_model_results()

        # Show dataframe in app with progress bars for quick comparisons
        _, col_res, _ = st.columns([0.2, 0.6, 0.2])
        col_res.dataframe(
            display_df,
            column_config={
                "Price": st.column_config.ProgressColumn(
                    "Price",
                    help="Estimated Price for Flight",
                    width="medium",
                    format="$%.2f",
                    min_value=display_df["Price"].min() * 0.95,
                    max_value=display_df["Price"].max() * 1.05,
                )
            },
            hide_index=True,
        )

    # Show flight path plot
    st.write("")
    fig = plot_flight_paths()
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
