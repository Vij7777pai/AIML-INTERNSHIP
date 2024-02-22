import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, r2_score

st.markdown("""
### About the Model
#### This model uses LSTM to predict tomorrow's Coca price based on the previous three days' Minimum, Maximum, and Modal prices.

:blossom::seedling::moneybag:

""")

def scale_dataframe(dataframe_dummy, selected_scaler):
    # Scale the data using the selected scaler
    columns_to_scale = ['Min', 'Max', 'Modal']
    dataframe_dummy[columns_to_scale] = selected_scaler.fit_transform(dataframe_dummy[columns_to_scale])
    return dataframe_dummy

def plot_trend(dataframe_dummy, title, x_label, y_label):
    st.title(title)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(dataframe_dummy.index, dataframe_dummy['Modal'])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    # Limit the number of x-axis date labels for better readability
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    plt.xticks(rotation=45)

    st.pyplot(fig)

def plot_predictions_streamlit(model, X, y, start=0, end=100, tag='Title'):
    pred = model.predict(X, verbose=0)
    df = pd.DataFrame({
        'predicted_Min': pred[:, 0],
        'Actual_Min': y[:, 0],
        'predicted_Max': pred[:, 1],
        'Actual_Max': y[:, 1],
        'predicted_Modal': pred[:, 2],
        'Actual_Modal': y[:, 2]
    })
    mae = mean_absolute_error(y, pred)
    #r2 = r2_score(y, pred)
    st.write(f'Mean absolute error for {tag} is {mae}')
    #st.write(f'r2 for {tag} is {r2*100}')

    # Plot Predicted and Actual Min
    fig, ax = plt.subplots(figsize=(20,6))
    ax.plot(df['predicted_Min'][start:end], label='Predicted_Min')
    ax.plot(df['Actual_Min'][start:end], label='Actual_Min')
    ax.set_title(tag)
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # Plot Predicted and Actual Max
    fig, ax = plt.subplots(figsize=(20,6))
    ax.plot(df['predicted_Max'][start:end], label='Predicted_Max')
    ax.plot(df['Actual_Max'][start:end], label='Actual_Max')
    ax.set_title(tag)
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # Plot Predicted and Actual Modal 
    fig, ax = plt.subplots(figsize=(20,6))
    ax.plot(df['predicted_Modal'][start:end], label='Predicted_Modal')
    ax.plot(df['Actual_Modal'][start:end], label='Actual_Modal')
    ax.set_title(tag)
    ax.legend(loc='upper left')
    st.pyplot(fig)


# Converting the dataframe to window
def df_to_X_y(df, window_size=1):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][0], df_as_np[i+window_size][1], df_as_np[i+window_size][2] 
        y.append(label)
    return np.array(X), np.array(y)



# Function to make predictions
def make_prediction(input):
    try:
        value_scaled = selected_scaler.transform(input) 
        value_scaled_reshaped = value_scaled.reshape(1, value_scaled.shape[0], value_scaled.shape[1])
        
        # Predict using the loaded model
        prediction = selected_model.predict(value_scaled_reshaped)
        
        prediction = selected_scaler.inverse_transform(prediction)
        
        return prediction[0]
    except ValueError:
        return None  

recursive_prediction = []
def make_and_store_predictions(input, num_days=10):
    # Clear the previous predictions
    recursive_prediction.clear()
    # Initial batch of input data
    last_batch = input
    # Make predictions for the specified number of days
    for i in range(num_days):
        next_day_prediction = make_prediction(last_batch)
        recursive_prediction.append(next_day_prediction)
        last_batch[-1] = next_day_prediction
    return recursive_prediction

def plot_future_prediction(recursive_prediction):
    st.title("Predictions for the Next 10 Days")
    
    # Create a list of days from 1 to 10
    days = list(range(1, 11))
    
    # Create subplots for each prediction
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    
    # Define colors for the lines
    colors = ['blue', 'green', 'red']
    
    # Plot each prediction (Minimum, Maximum, Modal)
    for i in range(3):
        ax = axes[i]
        ax.plot(days, [pred[i] for pred in recursive_prediction], marker='o', color=colors[i], label=f'Predicted {["Minimum", "Maximum", "Modal"][i]} Price')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        ax.set_title(f'Predicted {["Minimum", "Maximum", "Modal"][i]} Price for Next 10 Days')
        ax.grid(True)
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

# Example usage in Streamlit
if __name__ == '__main__':

    # Create a drop-down list to select the product
    selected_product = st.selectbox("Select a Product", ["Arecanut (Coca)", "Coconut (Grade-I)"])

    # Loading the LSTM model for the selected product
    if selected_product == "Arecanut (Coca)":
        model_path = './models/Coca/model_coca(3,3) 83.73.keras'
        scaler_path = './Scaler Objects/scaler_coca.pkl'
    elif selected_product == "Coconut (Grade-I)":
        model_path = './models/GradeI/model_gradeI(3,3)91.3473.keras'
        scaler_path = './Scaler Objects/scaler_grade-I.pkl'
        
    # Load the selected model and scaler
    selected_model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as scaler_file:
        selected_scaler = joblib.load(scaler_file)

    # Reading the CSV file for the selected product
    if selected_product == "Arecanut (Coca)":
        excel_coca = pd.read_excel('./Dataset/Coca/Coca_dataset.xlsx')
        dataframe = pd.DataFrame(excel_coca)
        dataframe = dataframe.set_index('Date')
    elif selected_product == "Coconut (Grade-I)":
        excel_gradeI = pd.read_excel('./Dataset/Grade-I/grade-I_test.xlsx')
        dataframe = pd.DataFrame(excel_gradeI)
        dataframe = dataframe.set_index('Date')

    if selected_product == 'Arecanut (Coca)':
        # Streamlit App
        st.title("Coca Price Predictor")
        st.subheader("Predict Tomorrow's Coca Price")
        st.write("Enter the previous three days' prices:")
        # Input fields for user
        # st.write("Enter the prices for the last three days:")
        col1, col2, col3 = st.columns(3)

        with col1:
            min_price_today = st.number_input("Minimum Price (Today)", value=22000.0, step=1.0)
    
        with col2:
            max_price_today = st.number_input("Maximum Price (Today)", value=28000.0, step=1.0)
    
        with col3:
            modal_price_today = st.number_input("Modal Price (Today)", value=24500.0, step=1.0)

        #st.write("Enter the prices for yesterday and the day before yesterday:")
        col4, col5, col6 = st.columns(3)

        with col4:
            min_price_yesterday = st.number_input("Minimum Price (Yesterday)", value=25000.0, step=1.0)
    
        with col5:
            max_price_yesterday = st.number_input("Maximum Price (Yesterday)", value=29264.0, step=1.0)
    
        with col6:
            modal_price_yesterday = st.number_input("Modal Price (Yesterday)", value=28177.0, step=1.0)
    
        col7, col8, col9 = st.columns(3)

        with col7:
            min_price_day_before = st.number_input("Minimum Price (Day Before Yesterday)", value=25000.0, step=1.0)
    
        with col8:
            max_price_day_before = st.number_input("Maximum Price (Day Before Yesterday)", value=29500.0, step=1.0)
    
        with col9:
            modal_price_day_before = st.number_input("Modal Price (Day Before Yesterday)", value=28000.0, step=1.0)

        input = [[min_price_today, max_price_today, modal_price_today],
         [min_price_yesterday, max_price_yesterday, modal_price_yesterday],
         [min_price_day_before, max_price_day_before, modal_price_day_before]]

        input_array = np.array(input)

        if st.button("Predict for tomorrow"):
            tom_pred = make_prediction(input_array)
            if len(tom_pred) >= 3:
                Min, Max, Modal = tom_pred[:3]  # Take the first 3 elements
                # Display the initial prediction
                st.write("Predicted Coca Price for Tomorrow:")
                st.write(f"Minimum Price: {Min:.3f}")
                st.write(f"Maximum Price: {Max:.3f}")
                st.write(f"Modal Price: {Modal:.3f}")

        # Always call make_and_store_predictions to calculate future predictions
        recursive_prediction = make_and_store_predictions(input_array, num_days=10)
    
        # Check if the "Future Predictions for Next 10 Days" button is clicked
        if st.button("Predictions for Next 10 Days") and len(recursive_prediction) >= 10:
            st.write("Predictions for the Next 10 Days:")
            for i, prediction in enumerate(recursive_prediction, start=1):
                st.write(f"Day {i} - Minimum Price: {prediction[0]:.4f}, Maximum Price: {prediction[1]:.3f}, Modal Price: {prediction[2]:.3f}")
    
            plot_future_prediction(recursive_prediction)
        
        plot_trend(dataframe, title='Arecanut (Coca) Trend', x_label='Date', y_label='Modal Price')

        dataframe = scale_dataframe(dataframe, selected_scaler)

        if st.button("Model Statistics"):
            #For coca
            WINDOW_SIZE = 3 
            X_coca, y_coca= df_to_X_y(dataframe,WINDOW_SIZE)
    
            X_train_coca, y_train_coca = X_coca[:1500], y_coca[:1500]
            X_val_coca, y_val_coca = X_coca[1500:1750], y_coca[1500:1750]
            X_test_coca, y_test_coca = X_coca[1750:] ,y_coca[1750:]

    
            # Create a Streamlit app
            st.title("Prediction Plots")
            st.header("Train and Plaidation Plots")
    
            # Display plots for training data
            st.subheader("Training Data")
            plot_predictions_streamlit(model=selected_model, X=X_train_coca, y=y_train_coca,start=0, end=len(X_train_coca), tag="Training")

            # Display plots for validation data
            st.subheader("Validation Data")
            plot_predictions_streamlit(model=selected_model, X=X_val_coca, y=y_val_coca, start=0, end=len(X_val_coca), tag="Training")

            # Display plots for test data
            st.subheader("Test Data")
            plot_predictions_streamlit(model=selected_model, X=X_test_coca, y=y_test_coca, start=0, end=len(X_test_coca), tag="Training")
    

    # For Coconut    
    if selected_product == 'Coconut (Grade-I)':
        
        # Streamlit App
        st.title("Coconut Price Predictor")
        st.subheader("Predict Tomorrow's Coconut Price")
        st.write("Enter the previous three days' prices:")
        # Input fields for user
        # st.write("Enter the prices for the last three days:")
        col1, col2, col3 = st.columns(3)

        with col1:
            min_price_today = st.number_input("Minimum Price (Today)", value=22000.0, step=1.0)
        
        with col2:
            max_price_today = st.number_input("Maximum Price (Today)", value=28000.0, step=1.0)
        
        with col3:
            modal_price_today = st.number_input("Modal Price (Today)", value=24500.0, step=1.0)

        #st.write("Enter the prices for yesterday and the day before yesterday:")
        col4, col5, col6 = st.columns(3)

        with col4:
            min_price_yesterday = st.number_input("Minimum Price (Yesterday)", value=25000.0, step=1.0)
        
        with col5:
            max_price_yesterday = st.number_input("Maximum Price (Yesterday)", value=29264.0, step=1.0)
        
        with col6:
            modal_price_yesterday = st.number_input("Modal Price (Yesterday)", value=28177.0, step=1.0)
        
        col7, col8, col9 = st.columns(3)

        with col7:
            min_price_day_before = st.number_input("Minimum Price (Day Before Yesterday)", value=25000.0, step=1.0)
        
        with col8:
            max_price_day_before = st.number_input("Maximum Price (Day Before Yesterday)", value=29500.0, step=1.0)
        
        with col9:
            modal_price_day_before = st.number_input("Modal Price (Day Before Yesterday)", value=28000.0, step=1.0)

        input = [[min_price_today, max_price_today, modal_price_today],
            [min_price_yesterday, max_price_yesterday, modal_price_yesterday],
            [min_price_day_before, max_price_day_before, modal_price_day_before]]

        input_array = np.array(input)

        if st.button("Predict for tomorrow"):
            tom_pred = make_prediction(input_array)
            if len(tom_pred) >= 3:
                Min, Max, Modal = tom_pred[:3]  # Take the first 3 elements
                # Display the initial prediction
                st.write("Predicted Coconut Price for Tomorrow:")
                st.write(f"Minimum Price: {Min:.3f}")
                st.write(f"Maximum Price: {Max:.3f}")
                st.write(f"Modal Price: {Modal:.3f}")

        # Always call make_and_store_predictions to calculate future predictions
        recursive_prediction = make_and_store_predictions(input_array, num_days=10)
        
        # Check if the "Future Predictions for Next 10 Days" button is clicked
        if st.button("Future Predictions for Next 10 Days") and len(recursive_prediction) >= 10:
            st.write("Future Predictions for the Next 10 Days:")
            for i, prediction in enumerate(recursive_prediction, start=1):
                st.write(f"Day {i} - Minimum Price: {prediction[0]:.4f}, Maximum Price: {prediction[1]:.3f}, Modal Price: {prediction[2]:.3f}")
        
            plot_future_prediction(recursive_prediction)
        

        plot_trend(dataframe, title='Coconut (Grade-I) Trend',x_label='Date', y_label='Modal Price')
        
        if st.button("Model Statistics"):
            dataframe = scale_dataframe(dataframe, selected_scaler)
            #For coca
            WINDOW_SIZE = 3 
            X_gradeI, y_gradeI= df_to_X_y(dataframe,WINDOW_SIZE)
        
            X_train_gradeI, y_train_gradeI = X_gradeI[:1300], y_gradeI[:1300]
            X_val_gradeI, y_val_gradeI = X_gradeI[1300:1400], y_gradeI[1300:1400]
            X_test_gradeI, y_test_gradeI = X_gradeI[1400:] ,y_gradeI[1400:] 

        
            # Create a Streamlit app
            st.title("Prediction Plots")
            st.header("Train and Vlaidation Plots")
        
            # Display plots for training data
            st.subheader("Training Data")
            plot_predictions_streamlit(model=selected_model, X=X_train_gradeI, y=y_train_gradeI,start=0, end=len(X_train_gradeI), tag="Training")

            # Display plots for validation data
            st.subheader("Validation Data")
            plot_predictions_streamlit(model=selected_model, X=X_val_gradeI, y=y_val_gradeI, start=0, end=len(X_val_gradeI), tag="Training")

            # Display plots for test data
            st.subheader("Test Data")
            plot_predictions_streamlit(model=selected_model, X=X_test_gradeI, y=y_test_gradeI, start=0, end=len(X_test_gradeI), tag="Training")




