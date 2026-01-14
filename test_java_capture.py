"""
Test script for Java SDK fingerprint capture integration

This script tests the complete pipeline:
1. Java helper captures fingerprint from ZKTeco scanner
2. Python API processes the image
3. CNN classification + Poincaré detection

Usage:
    python test_java_capture.py
"""

import subprocess
import base64
import sys
from pathlib import Path

def test_java_capture():
    """Test the Java capture helper directly"""
    print("=" * 60)
    print("Testing Java SDK Fingerprint Capture")
    print("=" * 60)
    print()
    
    # Check if Java is available
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("[OK] Java is installed")
        print(f"     Version: {result.stderr.splitlines()[0] if result.stderr else 'unknown'}")
    except Exception as e:
        print(f"[FAIL] Java not found: {e}")
        print("  Install JDK from https://adoptium.net/")
        return False
    
    print()
    
    # Check if files exist
    java_capture_dir = Path("java_ capture")
    capture_class = java_capture_dir / "CaptureOnce.class"
    zkfp_jar = Path("ZKFinger Standard SDK 5.3.0.33/Java/lib/ZKFingerReader.jar")
    native_dll = java_capture_dir / "libzkfp.dll"
    
    print("Checking required files:")
    files_ok = True
    
    if capture_class.exists():
        print(f"  [OK] {capture_class}")
    else:
        print(f"  [FAIL] {capture_class} (needs compilation)")
        files_ok = False
    
    if zkfp_jar.exists():
        print(f"  [OK] {zkfp_jar}")
    else:
        print(f"  [FAIL] {zkfp_jar} (missing)")
        files_ok = False
    
    if native_dll.exists():
        print(f"  [OK] {native_dll}")
    else:
        print(f"  [WARN] {native_dll} (optional - may work from system PATH)")
    
    print()
    
    if not files_ok:
        print("Please compile CaptureOnce.java first:")
        print(f'  cd "java_ capture"')
        print(f'  javac -cp "..\\{zkfp_jar}" CaptureOnce.java')
        return False
    
    # Test capture
    print("=" * 60)
    print("READY TO CAPTURE")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Make sure your ZKTeco fingerprint scanner is connected")
    print("2. Place your finger on the scanner when prompted")
    print("3. HOLD YOUR FINGER STILL for 5-10 seconds")
    print("4. Do NOT lift your finger until you see 'Capture successful'")
    print()
    input("Press ENTER to start capture...")
    print()
    
    try:
        # Build classpath (use absolute paths for Windows)
        import os
        zkfp_jar_abs = zkfp_jar.absolute()
        
        # Classpath includes current directory (.) and the JAR
        cp = f".;{zkfp_jar_abs}"
        
        # Set environment to include native library path
        env = os.environ.copy()
        java_capture_abs = java_capture_dir.absolute()
        env["PATH"] = f"{java_capture_abs};{env.get('PATH', '')}"
        
        print("Starting Java capture process...")
        print("(This will block until fingerprint is captured or timeout)")
        print(f"Classpath: {cp}")
        print(f"Working dir: {java_capture_abs}")
        print()
        
        result = subprocess.run(
            ["java", "-cp", cp, "CaptureOnce"],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            cwd=str(java_capture_abs),
            env=env
        )
        
        # Check result
        if result.returncode != 0:
            print(f"[FAIL] Capture failed (exit code {result.returncode})")
            print()
            print("STDOUT:")
            print(result.stdout)
            print()
            print("STDERR:")
            print(result.stderr)
            return False
        
        # Parse output
        output = result.stdout.strip()
        if not output.startswith("OK:"):
            print(f"[FAIL] Unexpected output: {output}")
            print()
            print("STDERR:")
            print(result.stderr)
            return False
        
        # Extract base64 data
        b64_data = output[3:].strip()
        
        try:
            img_bytes = base64.b64decode(b64_data)
            print(f"[SUCCESS] Capture successful!")
            print(f"          Image size: {len(img_bytes)} bytes")
            
            # Save test image
            test_output = Path("test_java_capture.bmp")
            test_output.write_bytes(img_bytes)
            print(f"          Saved to: {test_output}")
            print()
            
            print("=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("1. Start the API server: python api.py")
            print("2. Open http://localhost:8000/demo in your browser")
            print("3. Click 'Capture from Scanner' button")
            print()
            return True
            
        except Exception as e:
            print(f"[FAIL] Failed to decode image: {e}")
            return False
        
    except subprocess.TimeoutExpired:
        print("[FAIL] Timeout: No fingerprint detected after 60 seconds")
        print("       Make sure your finger is placed on the scanner")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_java_capture()
    sys.exit(0 if success else 1)

