
import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
import pandas as pd
import streamlit as st
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
import os
from datetime import datetime

st.set_page_config(
    page_title="🏨 Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Wider sidebar */
.css-1d391kg {
    background-color: #f0f8ff;
    padding: 1.5rem;
    border-radius: 10px;
    min-width: 350px !important;
}
/* Main content area */
.css-18e3th9 {
    padding: 2rem;
    background-color: #fafafa;
}
h1, h2, h3 {
    color: #5a67d8 !important;
    font-family: 'Arial Rounded MT Bold', sans-serif;
}
.stNumberInput>div>div>input, .stSelectbox>div>div>select {
    font-size: 16px !important;
    border-radius: 8px !important;
    border: 1px solid #cbd5e0 !important;
}
.highlight-box {
    background-color: #ebf8ff;
    border-left: 5px solid #4299e1;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.stButton>button {
    background-color: #667eea;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #5a67d8;
}
/* Big cancellation announcement */
.cancellation-announcement {
    font-size: 4rem !important;
    font-weight: bold;
    margin: 1rem 0;
    text-align: center;
}
/* Recommendation text */
.recommendation-text {
    color: #2d3748 !important;
    font-size: 1.3rem;
    margin: 1.5rem 0;
    text-align: center;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

class _RemainderColsList(list):
    pass

def load_model():
    """Load the trained model pipeline."""
    if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

    model_path = '/Users/ilham/Documents/final_project_draft/hotel_cancellation_model.pkl'
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error("❌ Model file not found. Please make sure 'hotel_cancellation_model.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"🛑 Error loading model: {str(e)}")
        return None

def user_input_features():
    """Collects user input features from the sidebar."""
    st.sidebar.header("🏨 Booking Details")
    
    with st.sidebar.expander("📅 Booking & Arrival Date", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            arrival_year = st.number_input("📅 Arrival Year", min_value=2023, max_value=2030, value=2024)
        with col2:
            arrival_month = st.selectbox("📅 Arrival Month", 
                ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December'], index=5)
        
        arrival_day = st.selectbox("📅 Arrival Day", options=list(range(1, 32)), index=14)
        lead_time = st.number_input("⏰ Lead Time (days)", min_value=0, max_value=365, value=30)
        
        month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        arrival_date = datetime(arrival_year, month_dict[arrival_month], arrival_day)
        arrival_date_week_number = arrival_date.isocalendar()[1]
    
    with st.sidebar.expander("💰 Pricing & Stay Details", expanded=True):
        adr = st.slider("💵 Average Daily Rate ($)", 50, 600, 120)
        total_nights = st.slider("🌙 Total Nights", 1, 30, 3)
        stays_in_weekend_nights = st.slider("🎉 Weekend Nights", 0, 10, 1)
        stays_in_week_nights = total_nights - stays_in_weekend_nights
    
    with st.sidebar.expander("👥 Guest Information", expanded=True):
        adults = st.number_input("👨‍👩‍👧‍👦 Adults", 1, 10, 2)
        children = st.number_input("🧒 Children", 0, 10, 0)
        babies = st.number_input("👶 Babies", 0, 5, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("🇺🇳 Country", 
                ['PRT', 'GBR', 'USA', 'ESP', 'FRA', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD', 'CHE'], index=0)
        with col2:
            customer_type = st.selectbox("👤 Customer Type", 
                ['Transient', 'Contract', 'Transient-Party', 'Group'], index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            previous_cancellations = st.number_input("❌ Previous Cancellations", 0, 10, 0)
        with col2:
            is_repeated_guest = st.selectbox("🔁 Repeat Guest", ['No', 'Yes'], index=0)
            is_repeated_guest = 1 if is_repeated_guest == 'Yes' else 0
    
    with st.sidebar.expander("🏢 Hotel & Room Details", expanded=True):
        hotel = st.selectbox("🏨 Hotel Type", ['City Hotel', 'Resort Hotel'], index=0)
        meal = st.selectbox("🍽️ Meal Plan", ['BB', 'FB', 'HB', 'SC', 'Undefined'], index=0)
        reserved_room_type = st.selectbox("🚪 Room Type", 
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'], index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            market_segment = st.selectbox("📊 Market Segment", 
                ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Undefined', 'Aviation'], index=2)
        with col2:
            distribution_channel = st.selectbox("📞 Distribution Channel", 
                ['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'], index=2)
        
        deposit_type = st.selectbox("💳 Deposit Type", 
            ['No Deposit', 'Non Refund', 'Refundable'], index=0)
    
    with st.sidebar.expander("⭐ Special Requests & Amenities", expanded=True):
        total_special_requests = st.slider("🎯 Special Requests", 0, 5, 1)
        required_car_parking_spaces = st.number_input("🅿️ Parking Spaces", 0, 5, 0)
        booking_changes = st.number_input("✏️ Booking Changes", 0, 10, 0)
        days_in_waiting_list = st.number_input("⏳ Days in Waitlist", 0, 100, 0)
    
    input_data = {
        'lead_time': [lead_time],
        'arrival_date_month': [arrival_month],
        'arrival_date_year': [arrival_year],
        'arrival_date_week_number': [arrival_date_week_number],
        'arrival_date_day_of_month': [arrival_day],
        'adr': [adr],
        'stays_in_weekend_nights': [stays_in_weekend_nights],
        'stays_in_week_nights': [stays_in_week_nights],
        'adults': [adults],
        'children': [children],
        'babies': [babies],
        'hotel': [hotel],
        'meal': [meal],
        'market_segment': [market_segment],
        'distribution_channel': [distribution_channel],
        'deposit_type': [deposit_type],
        'customer_type': [customer_type],
        'total_of_special_requests': [total_special_requests],
        'required_car_parking_spaces': [required_car_parking_spaces],
        'country': [country],
        'previous_cancellations': [previous_cancellations],
        'is_repeated_guest': [is_repeated_guest],
        'booking_changes': [booking_changes],
        'reserved_room_type': [reserved_room_type],
        'days_in_waiting_list': [days_in_waiting_list]
    }
    
    return pd.DataFrame(input_data)

def main():
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.markdown("Predict cancellation risk and optimize your revenue strategy")
    
    model = load_model()
    if model is None:
        return
    
    df_customer = user_input_features()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <h2 style='color: #5a67d8;'>📈 Cancellation Prediction ✨</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📋 Booking Details")
        features_display = df_customer.transpose()
        features_display.columns = ['Value']
        st.dataframe(features_display, use_container_width=True, height=400)
    
    with col2:
        prediction_text = "Click 'Predict Cancellation Risk' to see prediction"
        recommendation = "Fill out the form and click predict to get results"
        
        if st.button("🚀 Predict Cancellation Risk", type="primary", use_container_width=True):
            try:
                prediction = model.predict(df_customer)[0]
                prediction_text = "🚨 CANCEL" if prediction == 1 else "✅ NOT CANCEL"
                
                # Simple recommendation based on prediction
                if prediction == 1:
                    recommendation = "Recommend requiring a deposit or credit card guarantee to secure this booking."
                else:
                    recommendation = "Standard booking procedures apply. Low cancellation risk detected."
            
            except Exception as e:
                prediction_text = "❌ PREDICTION ERROR"
                recommendation = "Please check your input values and try again."
        
        # Big cancellation announcement
        st.markdown(f"""
        <div class='highlight-box'>
            <div style='text-align: center;'>
                <h3 style='margin-top: 0; color: #2b6cb0;'>Prediction Result</h3>
                <h1 class='cancellation-announcement'>{prediction_text}</h1>
                <p class='recommendation-text'>{recommendation}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='margin-top: 1.5rem;'>
                <h3 style='color: #4a5568;'>💡 Disclaimer</h3>
                <ul style='color: #718096;'>
                    <li>This model is a predictive tool based on historical data</li>
                    <li>Actual outcomes can vary and are influenced by many factors</li>
                    <li>For precise valuation, consult with hotel management experts</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()