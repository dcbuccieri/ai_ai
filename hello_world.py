#!/usr/bin/env python3
"""
Hello World script for the AI Stock Trading Bot project.
This script tests that the project structure and dependencies are working correctly.
"""

def main():
    """Main function to run the hello world test."""
    print("=" * 50)
    print("AI Stock Trading Bot - Hello World!")
    print("=" * 50)
    
    # Test basic Python functionality
    print("Python is working correctly")
    
    # Test project structure
    import os
    import sys
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Test if we can import from our src package
    try:
        from src import __init__
        print("src package is accessible")
    except ImportError as e:
        print(f"src package import failed: {e}")
        return False
    
    # Test if we can import from subpackages
    try:
        from src.models import __init__
        from src.preprocessing import __init__
        from src.evaluation import __init__
        from src.utils import __init__
        print("All subpackages are accessible")
    except ImportError as e:
        print(f"Subpackage import failed: {e}")
        return False
    
    # Test basic dependencies (if installed)
    dependencies_to_test = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('sklearn', 'Machine learning'),
        ('matplotlib', 'Plotting'),
        ('yfinance', 'Stock data')
    ]
    
    print("\nTesting dependencies:")
    for package, description in dependencies_to_test:
        try:
            __import__(package)
            print(f"[OK] {package} - {description}")
        except ImportError:
            print(f"[MISSING] {package} - {description} (not installed)")
    
    print("=" * 50)
    print("Hello World test completed!")
    print("To install dependencies: pip install -r requirements.txt")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
