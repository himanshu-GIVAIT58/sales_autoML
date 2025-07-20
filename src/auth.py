import streamlit as st
from streamlit_oauth import OAuth2Component
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Keycloak Configuration ---
KEYCLOAK_AUTHORIZE_URL = os.getenv("KEYCLOAK_AUTHORIZE_URL")
KEYCLOAK_TOKEN_URL = os.getenv("KEYCLOAK_TOKEN_URL")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET") # No default for secrets
REDIRECT_URI = os.getenv("REDIRECT_URI")

# --- CRITICAL: Check if secrets are loaded ---
if not all([KEYCLOAK_AUTHORIZE_URL, KEYCLOAK_TOKEN_URL, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET, REDIRECT_URI]):
    st.error("One or more Keycloak environment variables are not set. Please check your .env file.")
    st.stop()

# --- Create OAuth2 Component ---
oauth2 = OAuth2Component(
    client_id=KEYCLOAK_CLIENT_ID,
    client_secret=KEYCLOAK_CLIENT_SECRET,
    authorize_endpoint=KEYCLOAK_AUTHORIZE_URL,
    token_endpoint=KEYCLOAK_TOKEN_URL,
    refresh_token_endpoint=None,
    revoke_token_endpoint=None,
)

def check_authentication():
    """
    Checks if a user is authenticated. If not, it displays a login button.
    Returns the user's token if authenticated, otherwise returns None.
    """
    if 'token' not in st.session_state:
        result = oauth2.authorize_button(
            name="Login with Keycloak",
            icon="https://www.keycloak.org/resources/images/keycloak_icon_512px.png",
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="keycloak",
            use_container_width=True
        )
        if result and "token" in result:
            st.session_state.token = result.get("token")
            st.rerun()
        return None
    
    return st.session_state.token
