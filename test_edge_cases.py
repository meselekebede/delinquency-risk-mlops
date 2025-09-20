# test_edge_cases.py
import requests
import json
from typing import List, Dict

BASE_URL = "http://127.0.0.1:8000"

def test_case(name: str, payload: Dict, expected_status: int, contains: str = None):
    print(f"\n🧪 Test: {name}")
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"   Status: {response.status_code} (Expected: {expected_status})")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")

        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}"

        if contains:
            resp_text = json.dumps(response.json()).lower()
            assert contains.lower() in resp_text, f"Expected '{contains}' in response"
            print(f"   ✅ Contains '{contains}'")

        print("   ✅ PASSED")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Advanced Edge Case Tests...\n")

    # List to track results
    results = []

    # ✅ Valid Case – Should Pass
    valid_payload = {
        "age": 35,
        "gender": "Female",
        "education": "Tertiary",
        "employment_status": "Employed",
        "monthly_income": 20000,
        "loan_amount": 50000,
        "loan_tenure_months": 12,
        "loan_purpose": "Education",
        "region": "Urban",
        "previous_default": 0,
        "num_previous_loans": 2,
        "credit_score": 700
    }
    results.append(test_case("Valid Low-Risk Applicant", valid_payload, 200))

    # 🔴 Invalid Gender
    results.append(test_case(
        "Invalid Gender",
        {**valid_payload, "gender": "Other"},
        422,
        "must be one of"
    ))

    # 🔴 Negative Income
    results.append(test_case(
        "Negative Income",
        {**valid_payload, "monthly_income": -5000},
        422,
        "greater than 0"
    ))

    # 🔴 Age Out of Range
    results.append(test_case(
        "Underage Applicant",
        {**valid_payload, "age": 16},
        422,
        "greater than or equal to 18"
    ))

    # 🔴 Unknown Loan Purpose
    results.append(test_case(
        "Unknown Loan Purpose",
        {**valid_payload, "loan_purpose": "Vacation"},
        422,
        "must be one of"
    ))

    # 🔴 Extreme Loan-to-Income Ratio
    extreme_payload = {**valid_payload, "loan_amount": 500000, "monthly_income": 8000}
    result = requests.post(f"{BASE_URL}/predict", json=extreme_payload)
    if result.status_code == 200:
        risk = result.json()["delinquency_risk"]
        print(f"📈 Extreme case: High loan/income → Risk = {risk}")
        assert risk > 0.8, "Extreme ratio should yield high risk"
        results.append(True)
    else:
        print(f"❌ Failed with status {result.status_code}")
        results.append(False)

    # 🔴 Missing Required Field
    missing_payload = valid_payload.copy()
    missing_payload.pop("loan_amount")
    results.append(test_case(
        "Missing Loan Amount",
        missing_payload,
        422,
        "field required"
    ))

    # 🔴 Batch Stress Test (simulate multiple calls)
    print("\n🔥 Stress Test: Sending 10 rapid requests...")
    success_count = 0
    for i in range(10):
        try:
            response = requests.post(f"{BASE_URL}/predict", json=valid_payload)
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    print(f"   ✅ {success_count}/10 successful responses")
    results.append(success_count >= 8)

    # 📊 Summary
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"✅ SUMMARY: {passed}/{total} tests passed")
    print("="*50)

    if passed == total:
        print("🎉 All tests passed! Your API is robust.")
    else:
        print("⚠️  Some tests failed. Review error messages above.")