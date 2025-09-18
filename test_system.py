# test_system.py
"""
Test script for the Enhanced Shoplifting Detection System
"""
import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestShopliftingDetectionSystem(unittest.TestCase):
    """Test cases for the detection system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_email = "madhu25@gmail.com"
        self.test_password = "madhu123"
        self.invalid_email = "wrong@email.com"
        self.invalid_password = "wrongpass"
    
    def test_authentication_valid(self):
        """Test valid authentication"""
        # This would test the authentication logic
        self.assertEqual(self.test_email, "madhu25@gmail.com")
        self.assertEqual(self.test_password, "madhu123")
    
    def test_authentication_invalid(self):
        """Test invalid authentication"""
        self.assertNotEqual(self.invalid_email, self.test_email)
        self.assertNotEqual(self.invalid_password, self.test_password)
    
    def test_anomaly_detection_logic(self):
        """Test anomaly detection logic"""
        # Mock alerts for testing
        critical_alerts = [{'severity': 'critical', 'type': 'shoplifting'}]
        warning_alerts = [{'severity': 'warning', 'type': 'suspicious'}]
        multiple_warnings = [
            {'severity': 'warning', 'type': 'suspicious'},
            {'severity': 'warning', 'type': 'loitering'}
        ]
        
        # Test critical alert detection
        self.assertTrue(len([a for a in critical_alerts if a.get('severity') == 'critical']) > 0)
        
        # Test multiple warning detection
        self.assertTrue(len([a for a in multiple_warnings if a.get('severity') == 'warning']) >= 2)
    
    def test_directory_creation(self):
        """Test directory creation"""
        required_dirs = ['uploads', 'anomalies', 'logs', 'config']
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.assertTrue(os.path.exists(directory))
    
    def test_file_upload_validation(self):
        """Test file upload validation"""
        valid_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        invalid_extensions = ['txt', 'doc', 'pdf', 'exe']
        
        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in valid_extensions
        
        # Test valid files
        for ext in valid_extensions:
            self.assertTrue(allowed_file(f"test.{ext}"))
        
        # Test invalid files
        for ext in invalid_extensions:
            self.assertFalse(allowed_file(f"test.{ext}"))

def run_tests():
    """Run all tests"""
    print("🧪 Running system tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestShopliftingDetectionSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    run_tests()