"""Test script to check all imports in the modular structure"""
import os
import sys
import traceback

def test_import(module_path):
    try:
        print(f"Trying to import {module_path}...")
        __import__(module_path)
        print(f"✓ Successfully imported {module_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {module_path}: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test core imports
    modules = [
        "domain_classifier",
        "domain_classifier.api",
        "domain_classifier.api.app",
        "domain_classifier.classifiers",
        "domain_classifier.utils",
        "domain_classifier.storage",
        "domain_classifier.crawlers",
        "domain_classifier.config"
    ]
    
    success_count = 0
    for module in modules:
        if test_import(module):
            success_count += 1
    
    print(f"\nResults: {success_count}/{len(modules)} imports successful")
    
    # Check file existence
    core_files = [
        "domain_classifier/__init__.py",
        "domain_classifier/api/__init__.py", 
        "domain_classifier/api/app.py",
        "domain_classifier/classifiers/__init__.py",
        "domain_classifier/utils/__init__.py",
    ]
    
    print("\nChecking for required files:")
    for filepath in core_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath} exists")
        else:
            print(f"✗ {filepath} is missing")
