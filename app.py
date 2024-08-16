import os
import base64
import streamlit as st

from src.train import model  # Import the trained model from src.train

# Cache the inference function to optimize performance and avoid recomputation
@st.cache_data(max_entries=1000)
def inference_and_display_result(text):
    # Make a prediction using the trained model
    clf_prediction = model.predict(text)
    
    # Display the classification result
    st.markdown('**Classification Result**')
    st.write(f'Input: {text}')
    st.write(f'Prediction: {clf_prediction}')

# Main function to set up the Streamlit app
def main():
    st.set_page_config(
        page_title="Message Classification App",  
        page_icon=":envelope_with_arrow:",        
        layout="wide",                            
        menu_items={
            'Get Help': 'https://github.com/PUVHAM/Message_Classification',  
            'Report a Bug': 'mailto:phamquangvu19082005@gmail.com',        
            'About': "# Message Classification App\n"
                     "This app allows you to classify messages as spam or not spam using Naive Bayes algorithm."
        }
    )

    st.title(':sunflower: :blue[Naive Bayes] Message Classification Demo')

    uploaded_msg = st.text_input('Input your message (Press enter to run)', placeholder='Hi PUVHAM...')

    st.divider()  

    # Button to use an example message
    example_button = st.button('Use an example')

    # Display result using the example message or user input
    if example_button:
        inference_and_display_result('I love you')  
    elif uploaded_msg:
        inference_and_display_result(uploaded_msg)  

if __name__ == '__main__':
    main()
