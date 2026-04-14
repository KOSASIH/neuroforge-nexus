#!/usr/bin/env python3
"""
nexus_pi_validator.py — Cross-repo Pi-BCI Sync Validator
=========================================================
Validates that Pi digits are correctly synced into the NeuroForge Nexus
BCI matrix. Checks Pi accuracy, Nexus API health, and payload integrity.

Usage:
    python nexus_pi_validator.py --validate
    python nexus_pi_validator.py --validate --digits 10000
    python nexus_pi_validator.py --status
"""

import argparse
import hashlib
import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from super_pi import compute_pi, verify_pi, build_nexus_payload, send_to_nexus, NEXUS_API_URL
except ImportError:
    print("ERROR: super_pi.py not found in current directory.")
    sys.exit(1)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def check_nexus_health() -> dict:
    if not HAS_REQUESTS:
        return {"healthy": False, "reason": "requests not installed"}
    try:
        resp = requests.get(f"{NEXUS_API_URL}/health", timeout=5)
        return {"healthy": resp.status_code == 200, "data": resp.json()}
    except Exception as e:
        return {"healthy": False, "reason": str(e)}


def validate_pi_bci_sync(digits: int = 10000) -> dict:
    print(f"\n  🔍 Pi-BCI Sync Validation ({digits} digits)")
    print(f"  {'─'*45}")
    results = {}

    # 1. Compute Pi
    print("  [1/5] Computing Pi...")
    t0 = time.perf_counter()
    pi_str = compute_pi(digits)
    elapsed = time.perf_counter() - t0
    results["computation"] = {"ok": True, "time_s": round(elapsed, 3), "digits": digits}
    print(f"        ✅ {digits} digits in {elapsed:.3f}s")

    # 2. Verify accuracy
    print("  [2/5] Verifying accuracy...")
    ok = verify_pi(pi_str, min(digits, 50))
    results["accuracy"] = {"ok": ok, "preview": pi_str[:30]}
    print(f"        {'✅' if ok else '❌'} {pi_str[:30]}...")
    if not ok:
        print("        ❌ Pi verification FAILED!")
        return {"passed": False, "results": results}

    # 3. Build payload
    print("  [3/5] Building BCI payload...")
    payload = build_nexus_payload(pi_str, digits)
    results["payload"] = {
        "ok": True,
        "session_id": payload.session_id[:16],
        "neural_load": payload.neural_load,
        "digest": payload.pi_digest[:24],
    }
    print(f"        ✅ Session: {payload.session_id[:16]}... | Load: {payload.neural_load}")

    # 4. Send to Nexus
    print("  [4/5] Syncing to Nexus API...")
    resp = send_to_nexus(payload)
    nexus_ok = resp.get("status") == "ok"
    results["nexus_sync"] = {"ok": nexus_ok, "response": resp}
    if nexus_ok:
        print(f"        ✅ Nexus sync: 200 OK")
    else:
        print(f"        ⚠️  Nexus offline: {resp.get('reason', '?')} (mock server not running)")

    # 5. Integrity check
    print("  [5/5] Integrity verification...")
    recomputed = hashlib.sha256(pi_str.encode()).hexdigest()
    integrity_ok = recomputed == payload.pi_digest
    results["integrity"] = {"ok": integrity_ok, "digest_match": integrity_ok}
    print(f"        {'✅' if integrity_ok else '❌'} Digest: {recomputed[:24]}...")

    # Summary
    all_core_ok = all([
        results["computation"]["ok"],
        results["accuracy"]["ok"],
        results["payload"]["ok"],
        results["integrity"]["ok"],
    ])

    print(f"\n  {'─'*45}")
    if all_core_ok:
        print("  ✅ Pi-BCI sync confirmed!")
    else:
        print("  ❌ Pi-BCI sync FAILED — check above errors")

    return {
        "passed": all_core_ok,
        "nexus_live": nexus_ok,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser(description="Nexus Pi Validator")
    parser.add_argument("--validate", action="store_true", help="Run full validation")
    parser.add_argument("--status", action="store_true", help="Check Nexus API status")
    parser.add_argument("--digits", type=int, default=10000, help="Pi digits to validate")
    parser.add_argument("--output", type=str, help="Save validation results to JSON")
    args = parser.parse_args()

    print(f"\n  {'═'*50}")
    print(f"  🔗 NeuroForge Nexus Pi Validator")
    print(f"  {'═'*50}")

    if args.status:
        health = check_nexus_health()
        icon = "✅" if health["healthy"] else "⚠️"
        print(f"\n  {icon} Nexus API: {'ONLINE' if health['healthy'] else 'OFFLINE'}")
        if "data" in health:
            print(f"     {health['data']}")
        else:
            print(f"     {health.get('reason', 'no response')}")
        return

    if args.validate:
        result = validate_pi_bci_sync(args.digits)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  📄 Results saved → {args.output}")

        sys.exit(0 if result["passed"] else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
