import requests
import os
from urllib.parse import urlencode
from datetime import datetime, timedelta

class KeycloakClient:
    """Generic Keycloak client for authentication and token management"""
    
    def __init__(self, keycloak_url, realm, client_id, redirect_uri=None):
        """
        Initialize Keycloak client
        
        Args:
            keycloak_url (str): Base URL of Keycloak server
            realm (str): Keycloak realm name
            client_id (str): Client ID for the application
            redirect_uri (str): Redirect URI for OAuth flow
        """
        self.keycloak_url = keycloak_url.rstrip('/')
        self.realm = realm
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        
        # Construct URLs
        self.auth_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/auth"
        self.token_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token"
        self.userinfo_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/userinfo"
        self.logout_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/logout"
    
    def get_auth_url(self, state=None, scope='openid profile email'):
        """
        Generate authorization URL for OAuth flow
        
        Args:
            state (str): State parameter for CSRF protection
            scope (str): OAuth scope
            
        Returns:
            str: Authorization URL
        """
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': scope,
            'state': state or 'default_state'
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, code):
        """
        Exchange authorization code for access token
        
        Args:
            code (str): Authorization code from callback
            
        Returns:
            dict: Token response or None if failed
        """
        token_data = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'code': code,
            'redirect_uri': self.redirect_uri
        }
        
        try:
            response = requests.post(self.token_url, data=token_data)
            if response.status_code == 200:
                token_info = response.json()
                # Add expiration timestamp
                if 'expires_in' in token_info:
                    token_info['expires_at'] = datetime.now().timestamp() * 1000 + (token_info['expires_in'] * 1000)
                return token_info
            else:
                return None
        except Exception:
            return None
    
    def get_user_info(self, access_token):
        """
        Get user information using access token
        
        Args:
            access_token (str): Valid access token
            
        Returns:
            dict: User information or None if failed
        """
        headers = {'Authorization': f'Bearer {access_token}'}
        
        try:
            response = requests.get(self.userinfo_url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception:
            return None
    
    def refresh_token(self, refresh_token):
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token (str): Valid refresh token
            
        Returns:
            dict: New token information or None if failed
        """
        refresh_data = {
            'grant_type': 'refresh_token',
            'client_id': self.client_id,
            'refresh_token': refresh_token
        }
        
        try:
            response = requests.post(self.token_url, data=refresh_data)
            if response.status_code == 200:
                token_info = response.json()
                # Add expiration timestamp
                if 'expires_in' in token_info:
                    token_info['expires_at'] = datetime.now().timestamp() * 1000 + (token_info['expires_in'] * 1000)
                return token_info
            else:
                return None
        except Exception:
            return None
    
    def logout(self, refresh_token):
        """
        Logout user by invalidating refresh token
        
        Args:
            refresh_token (str): Refresh token to invalidate
            
        Returns:
            bool: True if logout successful, False otherwise
        """
        logout_params = {
            'client_id': self.client_id,
            'refresh_token': refresh_token
        }
        
        try:
            response = requests.post(self.logout_url, data=logout_params)
            return response.status_code == 204
        except Exception:
            return False
    
    def is_token_valid(self, token_info):
        """
        Check if access token is still valid
        
        Args:
            token_info (dict): Token information containing expires_at
            
        Returns:
            bool: True if token is valid, False otherwise
        """
        if not token_info or 'expires_at' not in token_info:
            return False
        
        now = datetime.now().timestamp() * 1000
        return now < token_info['expires_at']
    
    def authenticate_user(self, code):
        """
        Complete authentication flow with code exchange and user info retrieval
        
        Args:
            code (str): Authorization code from callback
            
        Returns:
            dict: Authentication result with user_info and token_info, or None if failed
        """
        token_info = self.exchange_code_for_token(code)
        if not token_info:
            return None
        
        user_info = self.get_user_info(token_info['access_token'])
        if not user_info:
            return None
        
        return {
            'user_info': user_info,
            'token_info': token_info,
            'authenticated': True,
            'timestamp': datetime.now().isoformat()
        }

# Factory function for creating Keycloak client with default configuration
def create_keycloak_client(redirect_uri=None):
    """
    Create Keycloak client with default configuration
    
    Args:
        redirect_uri (str): Redirect URI, defaults to APP_BASE_URL environment variable
        
    Returns:
        KeycloakClient: Configured Keycloak client
    """
    keycloak_url = os.getenv("KEYCLOAK_URL", "https://auth.givadiva.co")
    realm = os.getenv("KEYCLOAK_REALM", "dev")
    client_id = os.getenv("KEYCLOAK_CLIENT_ID", "cats-eye")
    
    if redirect_uri is None:
        base_url = os.getenv("APP_BASE_URL", "http://localhost:8501")
        redirect_uri = base_url
    
    return KeycloakClient(keycloak_url, realm, client_id, redirect_uri) 
