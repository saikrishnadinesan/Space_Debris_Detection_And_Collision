"""
Space-Track Login — spacetrack_csrf_cookie Fix
Run in VS Code Terminal: python fix_login.py
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("SPACETRACK_USER", "")
PASSWORD = os.getenv("SPACETRACK_PASS", "")

print(f"Username: {USERNAME}")
print(f"Password: {'*' * len(PASSWORD)}")
print()

BASE_URL   = "https://www.space-track.org"
LOGIN_PAGE = f"{BASE_URL}/auth/login"
LOGIN_URL  = f"{BASE_URL}/ajaxauth/login"

session = requests.Session()

# Step 1 — Get session cookies
print("📄 Step 1: Getting session cookies...")
session.get(LOGIN_PAGE, timeout=30, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
})

# Use the correct cookie name: spacetrack_csrf_cookie
csrf_token = session.cookies.get("spacetrack_csrf_cookie", "")
print(f"   CSRF token: {csrf_token[:20]}..." if csrf_token else "   No CSRF token")

# Step 2 — Login with correct CSRF token
print("\n🔐 Step 2: Logging in...")
resp = session.post(
    LOGIN_URL,
    data={
        "identity": USERNAME,
        "password": PASSWORD,
        "csrfmiddlewaretoken": csrf_token,
    },
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": LOGIN_PAGE,
        "Origin": BASE_URL,
        "X-CSRFToken": csrf_token,
        "X-Requested-With": "XMLHttpRequest",
    },
    allow_redirects=True,
    timeout=30
)

print(f"   Status   : {resp.status_code}")
print(f"   Response : {repr(resp.text[:200])}")

# Step 3 — Test query
print("\n🔍 Step 3: Testing authenticated query (ISS TLE)...")
test = session.get(
    f"{BASE_URL}/basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/json/limit/1",
    timeout=30
)
print(f"   Status   : {test.status_code}")
print(f"   Response : {test.text[:400]}")

if test.status_code == 200 and "[" in test.text:
    print("\n✅ LOGIN SUCCESSFUL!")

    # Now try downloading debris data
    print("\n📡 Downloading sample debris data...")
    debris = session.get(
        f"{BASE_URL}/basicspacedata/query/class/gp/OBJECT_TYPE/DEBRIS/orderby/NORAD_CAT_ID/limit/100/format/tle",
        timeout=60
    )
    print(f"   Debris query status  : {debris.status_code}")
    print(f"   Debris data preview  : {debris.text[:300]}")
else:
    print("\n❌ Login still failing.")
    print("\n💡 Please try these manual steps:")
    print("   1. Open browser → go to https://www.space-track.org")
    print("   2. Login manually — does it work?")
    print("   3. If yes: try resetting password and update .env")
    print("   4. If no: account may need approval (wait 24-48hrs)")
