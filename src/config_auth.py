"""
Keycloak Authentication Configuration

This file contains the configuration settings for Keycloak authentication.
Make sure to set these environment variables in your .env file or system environment.
"""

import os
from typing import Optional

# Keycloak Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "https://auth.givadiva.co")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "dev")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "cats-eye")

# Application Configuration
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")

# Cookie Configuration
COOKIE_EXPIRES_HOURS = int(os.getenv("COOKIE_EXPIRES_HOURS", "24"))

def validate_auth_config() -> tuple[bool, list[str]]:
    """
    Validate that all required authentication configuration is present.
    
    Returns:
        tuple: (is_valid, list_of_missing_vars)
    """
    required_vars = [
        "KEYCLOAK_URL",
        "KEYCLOAK_REALM", 
        "KEYCLOAK_CLIENT_ID"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars

def get_auth_config_summary() -> dict:
    """
    Get a summary of the current authentication configuration.
    
    Returns:
        dict: Configuration summary
    """
    return {
        "keycloak_url": KEYCLOAK_URL,
        "keycloak_realm": KEYCLOAK_REALM,
        "keycloak_client_id": KEYCLOAK_CLIENT_ID,
        "app_base_url": APP_BASE_URL,
        "cookie_expires_hours": COOKIE_EXPIRES_HOURS
    }

def print_auth_setup_instructions():
    """
    Print setup instructions for Keycloak authentication.
    """
    print("=" * 60)
    print("KEYCLOAK AUTHENTICATION SETUP")
    print("=" * 60)
    print()
    print("Required Environment Variables:")
    print("1. KEYCLOAK_URL - Your Keycloak server URL")
    print("2. KEYCLOAK_REALM - The realm name")
    print("3. KEYCLOAK_CLIENT_ID - Your client ID")
    print("4. APP_BASE_URL - Your application base URL (optional)")
    print()
    print("Example .env file:")
    print("KEYCLOAK_URL=https://auth.givadiva.co")
    print("KEYCLOAK_REALM=dev")
    print("KEYCLOAK_CLIENT_ID=cats-eye")
    print("APP_BASE_URL=http://localhost:8501")
    print()
    print("Keycloak Client Configuration:")
    print("1. Create a new client in your Keycloak realm")
    print("2. Set client protocol to 'openid-connect'")
    print("3. Set access type to 'public'")
    print("4. Add your redirect URI: {APP_BASE_URL}")
    print("5. Enable 'Standard Flow' in client settings")
    print("6. Add 'openid', 'profile', 'email' to Valid Redirect URIs")
    print()
    print("For more information, see the Keycloak documentation.")
    print("=" * 60)

if __name__ == "__main__":
    # Validate configuration
    is_valid, missing = validate_auth_config()
    
    if not is_valid:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print()
        print_auth_setup_instructions()
    else:
        print("✅ Authentication configuration is valid!")
        print()
        print("Current configuration:")
        config = get_auth_config_summary()
        for key, value in config.items():
            print(f"   {key}: {value}") 