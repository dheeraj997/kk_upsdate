import streamlit as st
import multipage_streamlit as mt
from StockDashboard import StockDashboard
from model import runn

def main():
    st.title("My Streamlit App")

    # Create a container for the header
    header_container = st.container()

    # Inside the header container, create the navigation bar
    with header_container:
        st.title("Navigation")

        # Create buttons for navigation
        if st.button("Home"):
            st.session_state.page = "home"

        if st.button("Run Model"):
            st.session_state.page = "run"

    # Get the current page from the session state
    current_page = st.session_state.get("page", "home")

    # Render the appropriate page based on the value of 'page'
    if current_page == "home":
        dashboard = StockDashboard()
        dashboard.run()  # Call the main function of the homepage module
    elif current_page == "run":
        runn()  # Call the run function from the model module
        
if __name__ == "__main__":
    main()
