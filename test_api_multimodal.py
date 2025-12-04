"""
Test script for the /diagnosticar_com_descricao API endpoint
Requires Flask app running on port 5001
"""

import requests


def test_multimodal_endpoint():
    """Test the multimodal diagnosis endpoint"""
    api_url = "http://localhost:5001/diagnosticar_com_descricao"

    test_cases = [
        {
            "name": "Pneumonia Case",
            "file": "imgs/person75_bacteria_365.jpeg",
            "expected_class": "PNEUMONIA",
        },
        {
            "name": "Normal Case",
            "file": "imgs/1_normal1.jpeg",
            "expected_class": "NORMAL",
        },
    ]

    print("=" * 80)
    print("MULTIMODAL API ENDPOINT TEST")
    print(f"Testing: POST {api_url}")
    print("=" * 80)

    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {idx}: {test_case['name']}")
        print(f"File: {test_case['file']}")
        print("=" * 80)

        try:
            # Open and send image
            with open(test_case["file"], "rb") as img_file:
                files = {"imagem": img_file}
                response = requests.post(api_url, files=files, timeout=60)

            # Check response
            print(f"\n📡 HTTP Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                print(f"\n📊 DIAGNOSIS RESULT:")
                print(f"   Class: {data.get('class')}")
                print(f"   Confidence: {data.get('confidence'):.1%}")
                print(
                    f"   Expected: {test_case['expected_class']}"
                    f" {'✅' if data.get('class') == test_case['expected_class'] else '❌'}"
                )

                print(f"\n🖼️  VISUALIZATION URLS:")
                print(f"   Heatmap: {data.get('heatmap_url')}")
                print(f"   Overlay: {data.get('overlay_url')}")

                print(f"\n🤖 LLM CLINICAL DESCRIPTION:")
                print("-" * 80)
                print(data.get("description"))
                print("-" * 80)

            else:
                print(f"\n❌ Error: {response.text}")

        except requests.exceptions.ConnectionError:
            print("\n❌ Connection Error: Flask app not running")
            print("   Start the app with: mise run run")
            return

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_multimodal_endpoint()
