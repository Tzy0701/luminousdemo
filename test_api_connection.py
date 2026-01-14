"""
Test API Connection to Render Backend
Quick script to verify your backend is accessible
"""

import requests
import sys

API_URL = "https://luminousproject.onrender.com"

def test_connection():
    print("=" * 70)
    print("Testing Luminous API Connection")
    print("=" * 70)
    print(f"\nTarget: {API_URL}\n")
    
    # Test 1: Root endpoint
    print("Test 1: Root endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code == 200:
            print(f"   [OK] SUCCESS - Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        else:
            print(f"   [WARN] WARNING - Status: {response.status_code}")
    except requests.exceptions.Timeout:
        print("   [TIMEOUT] Backend may be sleeping (Render free tier)")
        print("   [TIP] Try again in 50 seconds")
        return False
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False
    
    # Test 2: Health endpoint
    print("\nTest 2: Health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            print(f"   [OK] SUCCESS - Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        else:
            print(f"   [WARN] WARNING - Status: {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False
    
    # Test 3: Demo endpoint
    print("\nTest 3: Demo endpoint...")
    try:
        response = requests.get(f"{API_URL}/demo", timeout=10)
        if response.status_code == 200:
            print(f"   [OK] SUCCESS - Demo page accessible")
        else:
            print(f"   [WARN] WARNING - Status: {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All tests passed! Your backend is working correctly.")
    print("=" * 70)
    print("\nNext steps:")
    print("   1. Configure Streamlit Cloud with API_URL secret")
    print("   2. Deploy your frontend")
    print("   3. Test fingerprint upload in the web interface")
    print()
    return True

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

