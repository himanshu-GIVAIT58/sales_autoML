#!/usr/bin/env python3
"""
Keycloak Authentication Test Script

This script helps you test your Keycloak authentication configuration.
Run this script to verify that your Keycloak setup is working correctly.
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config_auth import validate_auth_config, get_auth_config_summary, print_auth_setup_instructions
from src.utils.keycloak import create_keycloak_client

def test_keycloak_connection():
    """Test the connection to Keycloak server"""
    try:
        config = get_auth_config_summary()
        keycloak_url = config['keycloak_url']
        realm = config['keycloak_realm']
        
        # Test basic connection
        print(f"🔍 Testing connection to Keycloak server...")
        print(f"   URL: {keycloak_url}")
        print(f"   Realm: {realm}")
        
        # Test if the server is reachable
        response = requests.get(f"{keycloak_url}/realms/{realm}", timeout=10)
        
        if response.status_code == 200:
            print("✅ Keycloak server is reachable!")
            realm_info = response.json()
            print(f"   Realm name: {realm_info.get('realm', 'Unknown')}")
            print(f"   Realm enabled: {realm_info.get('enabled', False)}")
            return True
        else:
            print(f"❌ Keycloak server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Keycloak server. Check the URL and network connectivity.")
        return False
    except Exception as e:
        print(f"❌ Error testing Keycloak connection: {str(e)}")
        return False

def test_keycloak_client():
    """Test the Keycloak client initialization"""
    try:
        print("🔍 Testing Keycloak client initialization...")
        client = create_keycloak_client()
        print("✅ Keycloak client initialized successfully!")
        
        # Test auth URL generation
        auth_url = client.get_auth_url(state="test")
        print(f"✅ Auth URL generated: {auth_url[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing Keycloak client: {str(e)}")
        return False

def test_cookie_manager():
    """Test the cookie manager initialization"""
    try:
        print("🔍 Testing cookie manager initialization...")
        from src.utils.cookie_utils import create_auth_cookie_manager
        cookie_manager = create_auth_cookie_manager()
        print("✅ Cookie manager initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing cookie manager: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("KEYCLOAK AUTHENTICATION TEST")
    print("=" * 60)
    print()
    
    # Load environment variables
    load_dotenv()
    
    # Validate configuration
    is_valid, missing = validate_auth_config()
    
    if not is_valid:
        print("❌ Configuration validation failed!")
        print("Missing variables:")
        for var in missing:
            print(f"   - {var}")
        print()
        print_auth_setup_instructions()
        return False
    
    print("✅ Configuration validation passed!")
    print()
    
    # Show current configuration
    print("Current configuration:")
    config = get_auth_config_summary()
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Run tests
    tests = [
        ("Keycloak Connection", test_keycloak_connection),
        ("Keycloak Client", test_keycloak_client),
        ("Cookie Manager", test_cookie_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running test: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Your Keycloak authentication is ready to use.")
        print()
        print("Next steps:")
        print("1. Start your Streamlit application")
        print("2. Navigate to the dashboard")
        print("3. You should see the login page")
        print("4. Click 'Login with Keycloak' to test the authentication flow")
    else:
        print("❌ Some tests failed. Please check the configuration and try again.")
        print()
        print("Common issues:")
        print("- Check if Keycloak server is running and accessible")
        print("- Verify the realm name and client ID")
        print("- Ensure the client is configured for public access")
        print("- Check that redirect URIs are properly configured")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 