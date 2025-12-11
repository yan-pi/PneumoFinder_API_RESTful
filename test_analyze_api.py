#!/usr/bin/env python3
"""Test /analyze endpoint."""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:5001"

def test_health():
    """Test health endpoint."""
    print("🔍 Testing GET /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200
    print("   ✅ Health check passed\n")

def test_analyze(image_path: str, translate: bool = False):
    """Test /analyze endpoint."""
    url_suffix = "?translate=true" if translate else ""
    print(f"🔍 Testing POST /analyze{url_suffix}...")
    print(f"   Image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"   ❌ Image not found: {image_path}")
        sys.exit(1)
    
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{API_URL}/analyze{url_suffix}", files=files)
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Success!")
        print(f"   Diagnosis ID: {result.get('diagnosis_id')}")
        print(f"   CNN: {result['cnn_diagnosis']['prediction']} ({result['cnn_diagnosis']['confidence']:.2f})")
        print(f"   Total latency: {result['total_latency_s']:.1f}s")
        
        if 'stage1_vision' in result:
            print(f"   Stage 1: {result['stage1_vision']['latency_s']:.1f}s")
        if 'stage2_medical' in result:
            print(f"   Stage 2: {result['stage2_medical']['latency_s']:.1f}s")
        if 'stage3_translation' in result:
            print(f"   Stage 3: {result['stage3_translation']['latency_s']:.1f}s")
        
        print(f"\n   📄 Final Report (EN):")
        print(f"   {result['final_report_en'][:150]}...")
        
        if 'final_report_pt_br' in result:
            print(f"\n   📄 Final Report (PT-BR):")
            print(f"   {result['final_report_pt_br'][:150]}...")
        
        print()
    else:
        print(f"   ❌ Failed:")
        print(f"   {response.json()}\n")

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TESTING /analyze ENDPOINT")
    print("=" * 80)
    print()
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Analyze without translation (fast)
    test_analyze("imgs/person75_bacteria_365.jpeg", translate=False)
    
    # Optional: Test with translation (slow)
    # test_analyze("imgs/person75_bacteria_365.jpeg", translate=True)
    
    print("=" * 80)
    print("🏁 Tests Complete")
    print("=" * 80)
