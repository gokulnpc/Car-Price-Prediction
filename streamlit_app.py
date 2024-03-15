import streamlit as st
import pandas as pd
import pickle
import joblib

# Function to load the model
@st.cache_data
def load_model():
    with open('best_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

# Load your model
loaded_model = load_model()

# Function to create the input datafram
def create_input_df(user_inputs, category_map):
    data = {f'{category}_{sub_category}': 0 for category, sub_categories in category_map.items() for sub_category in sub_categories}
    # intialize the numerical columns
    data.update({col: 0 for col in ['engine', 'fuel tank capacity', 'kilometer', 'seating capacity', 'year']})
    input_df = pd.DataFrame(data, index=[0])
    # Update the input_df with the user inputs
    for category, value in user_inputs.items():
        if category in ['engine', 'fuel tank capacity', 'kilometer', 'seating capacity', 'year']:
            input_df[category] = value
        else:
            input_df[f'{category}_{value}'] = 1
    input_df.reset_index(drop=True, inplace=True)
    input_df = input_df.reindex(columns=sorted(input_df.columns))
    return input_df
    
# Define your category_map
category_map = {
    'color': ['Black', 'Blue', 'Bronze', 'Brown', 'Gold', 'Green', 'Grey', 'Maroon', 'Orange', 'Others', 'Pink', 'Purple', 'Red', 'Silver', 'White', 'Yellow'],
    'fuel type': ['CNG + CNG', 'Diesel', 'Electric', 'Hybrid', 'LPG', 'Petrol', 'Petrol + CNG', 'Petrol + LPG'],
    'make': ['BMW', 'Chevrolet', 'Datsun', 'Ferrari', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Lamborghini', 'Land Rover', 'Lexus', 'MG', 'MINI', 'Mahindra', 'Maruti Suzuki', 'Maserati', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Rolls-Royce', 'Skoda', 'Ssangyong', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'],
    'owner': ['First', 'Fourth', 'Second', 'Third', 'UnRegistered Car'],
    'seller type': ['Corporate', 'Individual'],
    'transmission': ['Manual', 'Automatic']
}
# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Car Price Prediction')

    color = st.selectbox('Select the car color:', category_map['color'])
    fuel_type = st.selectbox('Select the fuel type:', category_map['fuel type'])
    make = st.selectbox('Select the car make:', category_map['make'])
    owner = st.selectbox('Select the owner type:', category_map['owner'])
    seller_type = st.selectbox('Select the seller type:', category_map['seller type'])
    transmission = st.selectbox('Select the transmission type:', category_map['transmission'])
    year = st.number_input('Enter the year of the car:', min_value=2000, max_value=2022, value=2010)
    engine = st.number_input('Enter the engine capacity (cc):', min_value=500, max_value=5000, value=1000)
    fuel_tank = st.number_input('Enter the fuel tank capacity (litres):', min_value=10, max_value=100, value=30)
    seat = st.number_input('Enter the seating capacity:', min_value=2, max_value=10, value=5)
    km = st.number_input('Enter the kilometer run (in km):', min_value=0, max_value=100000, value=50000)

    user_inputs = {
            'color': color,
            'fuel type': fuel_type,
            'make': make,
            'owner': owner,
            'seller type': seller_type,
            'transmission': transmission,
            'year': year,
            'engine': engine,
            'fuel tank capacity': fuel_tank,
            'seating capacity': seat,
            'kilometer': km
    }

    
    # Create a button to predict the output
    if st.button('Predict'):
        input_df = create_input_df(user_inputs, category_map)
        prediction = loaded_model.predict(input_df)
        st.markdown(f'**The predicted price of the car is: {prediction[0]:,.2f}**') 

  
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(loaded_model.get_params())
            st.write('Model used: Random Forest Regressor')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'Car_Price_Prediction.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="Chronic Kidney Disease Prediction.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'car_data.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="car_data.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Car-Price-Prediction)')
    st.write('--'*50)

elif options == 'About':
    st.title('About')
    st.write('This is a simple web app to predict the price of a car based on the user inputs. The model used in this web app is a Random Forest Regressor model. The model was trained on a dataset containing information about used cars. The dataset was collected from the Kaggle website. The dataset contains the following columns:')
    st.write('1. Name: The name of the car')
    st.write('2. Year: The year in which the car was manufactured')
    st.write('3. Selling Price: The price at which the car was sold')
    st.write('4. Present Price: The current price of the car')
    st.write('5. Kms Driven: The number of kilometers the car has been driven')
    st.write('6. Fuel Type: The type of fuel used by the car')
    st.write('7. Seller Type: The type of seller (Individual or Dealer)')
    st.write('8. Transmission: The type of transmission (Manual or Automatic)')
    st.write('9. Owner: The number of previous owners of the car')
    st.write('10. Mileage: The mileage of the car')
    st.write('11. Engine: The engine capacity of the car')
    st.write('12. Power: The power of the car')
    st.write('13. Seats: The seating capacity of the car')
    st.write('14. Brand: The brand of the car')
    st.write('15. Model: The model of the car')
    st.write('16. Color: The color of the car')
    st.write('17. Fuel Tank Capacity: The capacity of the fuel tank')
    st.write('--'*50)
    


    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Car-Price-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)