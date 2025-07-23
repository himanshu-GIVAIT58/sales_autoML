import json
from datetime import datetime, timedelta
from extra_streamlit_components import CookieManager

class AuthCookieManager:
    """Utility class for managing authentication cookies"""
    
    def __init__(self, instance_id=None):
        # Create a unique key for this instance
        if instance_id is None:
            import uuid
            instance_id = str(uuid.uuid4())[:8]
        
        self.cookie_manager = CookieManager(key=f"auth_cookie_manager_{instance_id}")
        self.auth_cookie_key = 'auth_data'
    
    def save_auth_data(self, user_info, token_info, expires_hours=1):
        """Save authentication data to cookies"""
        # Add expiration time to token info
        if 'expires_in' in token_info:
            expires_at = datetime.now().timestamp() * 1000 + (token_info['expires_in'] * 1000)
            token_info['expires_at'] = expires_at
        
        auth_data = {
            'authenticated': True,
            'user_info': user_info,
            'token_info': token_info,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Set authentication cookie with expiration
            self.cookie_manager.set(
                self.auth_cookie_key,
                json.dumps(auth_data),
                expires_at=datetime.now() + timedelta(hours=expires_hours)
            )
            return True, "Authentication data saved to cookies!"
        except Exception as e:
            return False, f"Error saving to cookies: {str(e)}"
    
    def get_auth_data(self):
        """Retrieve authentication data from cookies"""
        try:
            auth_cookie = self.cookie_manager.get(self.auth_cookie_key)
            
            if auth_cookie:
                # If it's already a dict, use it directly
                if isinstance(auth_cookie, dict):
                    auth_data = auth_cookie
                else:
                    # If it's a string, parse it as JSON
                    auth_data = json.loads(auth_cookie)
                return True, auth_data
            else:
                return False, None
                
        except Exception as e:
            return False, f"Error retrieving from cookies: {str(e)}"
    
    def is_token_valid(self, auth_data):
        """Check if the authentication token is still valid"""
        if not auth_data or 'token_info' not in auth_data:
            return False
        
        token_info = auth_data['token_info']
        if 'expires_at' in token_info:
            expires_at = token_info['expires_at']
            now = datetime.now().timestamp() * 1000
            
            return now < expires_at
        
        # If no expiration info, assume valid for now
        return True
    
    def restore_auth_session(self):
        """Restore authentication session from cookies"""
        success, result = self.get_auth_data()
        
        if not success:
            return False, result
        
        if result and self.is_token_valid(result):
            # Return the auth data for external session management
            return True, result
        else:
            # Token expired or invalid
            self.clear_auth_cookies()
            return False, "Authentication token has expired"
    
    def clear_auth_cookies(self):
        """Clear authentication cookies"""
        try:
            self.cookie_manager.delete(self.auth_cookie_key)
            return True, "Authentication cookies cleared!"
        except Exception as e:
            return False, f"Error clearing cookies: {str(e)}"
    
    def get_all_cookies(self):
        """Get all cookies for debugging"""
        try:
            return self.cookie_manager.get_all()
        except Exception as e:
            return {"error": str(e)}
    
    def get_cookie_manager(self):
        """Get the cookie manager instance"""
        return self.cookie_manager

# Global instance to avoid duplicate CookieManager instances
_auth_cookie_manager_instance = None

def create_auth_cookie_manager():
    """Factory function to create an AuthCookieManager instance"""
    global _auth_cookie_manager_instance
    if _auth_cookie_manager_instance is None:
        _auth_cookie_manager_instance = AuthCookieManager(instance_id="main")
    return _auth_cookie_manager_instance

def save_auth_to_cookies(user_info, token_info, expires_hours=1):
    """Convenience function to save auth data to cookies"""
    manager = create_auth_cookie_manager()
    success, message = manager.save_auth_data(user_info, token_info, expires_hours)
    
    return success, message

def restore_auth_from_cookies():
    """Convenience function to restore auth from cookies"""
    manager = create_auth_cookie_manager()
    success, result = manager.restore_auth_session()
    
    return success, result

def clear_auth_cookies():
    """Convenience function to clear auth cookies"""
    manager = create_auth_cookie_manager()
    success, message = manager.clear_auth_cookies()
    
    return success, message

def get_auth_cookie_data():
    """Convenience function to get auth cookie data"""
    manager = create_auth_cookie_manager()
    success, result = manager.get_auth_data()
    
    if success and result:
        return result
    else:
        return None 
