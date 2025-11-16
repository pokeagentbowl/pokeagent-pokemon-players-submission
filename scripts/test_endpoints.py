#!/usr/bin/env python3
"""
Script to test all v3 server endpoints programmatically.
Loads the torchic state and verifies all endpoints work.
"""

import requests
import json
import time
import base64
import os
from pathlib import Path

class EndpointTester:
    def __init__(self, base_url="http://localhost:8000", output_dir="docs/endpoint_tests"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def wait_for_server(self, timeout=30):
        """Wait for server to be ready"""
        print("Waiting for server to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(1)
        print("❌ Server did not become ready in time")
        return False
    
    def test_endpoint(self, method, endpoint, name, save_response=True, **kwargs):
        """Test a single endpoint"""
        print(f"\nTesting {name}...")
        try:
            url = f"{self.base_url}{endpoint}"
            if method.upper() == "GET":
                response = requests.get(url, timeout=10, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=10, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            result = {
                "name": name,
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "timestamp": time.time()
            }
            
            # Try to parse JSON response
            try:
                result["response"] = response.json()
            except:
                result["response_text"] = response.text[:500] if len(response.text) > 500 else response.text
            
            # Save response to file if requested and not too large
            if save_response and response.status_code == 200:
                try:
                    data = response.json()
                    # Save JSON response (excluding large fields like screenshots)
                    clean_data = self._clean_for_save(data)
                    filename = self.output_dir / f"{name.replace(' ', '_').lower()}.json"
                    with open(filename, 'w') as f:
                        json.dump(clean_data, f, indent=2)
                    result["saved_to"] = str(filename)
                except:
                    pass
            
            self.results[name] = result
            status = "✅" if result["success"] else "❌"
            print(f"{status} {name}: {response.status_code}")
            return result
            
        except Exception as e:
            result = {
                "name": name,
                "endpoint": endpoint,
                "method": method,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
            self.results[name] = result
            print(f"❌ {name}: Error - {str(e)}")
            return result
    
    def _clean_for_save(self, data):
        """Remove or truncate large fields for saving"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if key.endswith("_base64") or key == "screenshot_base64":
                    cleaned[key] = f"<base64 data: {len(value)} chars>"
                elif isinstance(value, (dict, list)):
                    cleaned[key] = self._clean_for_save(value)
                else:
                    cleaned[key] = value
            return cleaned
        elif isinstance(data, list):
            return [self._clean_for_save(item) for item in data]
        else:
            return data
    
    def save_screenshot(self, screenshot_data, filename):
        """Save base64 screenshot to file"""
        try:
            img_data = base64.b64decode(screenshot_data)
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(img_data)
            print(f"💾 Saved screenshot to {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"❌ Failed to save screenshot: {e}")
            return None
    
    def test_all_endpoints(self):
        """Test all available endpoints"""
        print("=" * 60)
        print("Testing all v3 server endpoints")
        print("=" * 60)
        
        # Health/Status checks
        self.test_endpoint("GET", "/health", "Health Check")
        self.test_endpoint("GET", "/status", "Status")
        
        # Basic endpoints
        self.test_endpoint("GET", "/screenshot", "Screenshot")
        self.test_endpoint("GET", "/api/frame", "API Frame")
        
        # Comprehensive state
        state_result = self.test_endpoint("GET", "/state", "Comprehensive State")
        
        # Extract and save screenshot from state if available
        if state_result["success"] and "response" in state_result:
            state_data = state_result["response"]
            if "visual" in state_data and "screenshot_base64" in state_data["visual"]:
                self.save_screenshot(
                    state_data["visual"]["screenshot_base64"],
                    "initial_screenshot.png"
                )
        
        # Milestones
        self.test_endpoint("GET", "/milestones", "Milestones")
        
        # Agent and metrics
        self.test_endpoint("GET", "/agent", "Agent Status")
        self.test_endpoint("GET", "/metrics", "Metrics")
        self.test_endpoint("GET", "/recent_actions", "Recent Actions")
        
        # Queue status
        self.test_endpoint("GET", "/queue_status", "Queue Status")
        
        # Debug endpoints
        self.test_endpoint("GET", "/debug/memory", "Debug Memory Basic")
        self.test_endpoint("GET", "/debug/memory/comprehensive", "Debug Memory Comprehensive")
        self.test_endpoint("GET", "/debug/milestones", "Debug Milestones")
        
        # LLM logs
        self.test_endpoint("GET", "/llm_logs", "LLM Logs")
        
        print("\n" + "=" * 60)
        print("Endpoint testing complete!")
        print("=" * 60)
    
    def test_movement(self):
        """Test player movement by sending button inputs"""
        print("\n" + "=" * 60)
        print("Testing Player Movement")
        print("=" * 60)
        
        # Get initial position
        initial_state = self.test_endpoint("GET", "/state", "Get Initial State", save_response=False)
        if not initial_state["success"]:
            print("❌ Could not get initial state")
            return
        
        initial_pos = initial_state["response"]["player"]["position"]
        print(f"📍 Initial position: {initial_pos}")
        
        # Try moving in different directions
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        for direction in directions:
            print(f"\n🎮 Testing movement: {direction}")
            
            # Send 10 button presses in the same direction
            buttons = [direction] * 10
            response = self.test_endpoint(
                "POST", "/action", f"Move {direction}",
                save_response=False,
                json={"buttons": buttons}
            )
            
            if not response["success"]:
                print(f"❌ Failed to send {direction} action")
                continue
            
            # Wait for action to complete
            time.sleep(2)
            
            # Get new position
            new_state = self.test_endpoint("GET", "/state", f"Get State After {direction}", save_response=False)
            if new_state["success"]:
                new_pos = new_state["response"]["player"]["position"]
                print(f"📍 New position: {new_pos}")
                
                if new_pos != initial_pos:
                    print(f"✅ Player moved! Direction {direction} works!")
                    
                    # Save screenshot and map for successful movement
                    if "visual" in new_state["response"] and "screenshot_base64" in new_state["response"]["visual"]:
                        self.save_screenshot(
                            new_state["response"]["visual"]["screenshot_base64"],
                            f"after_move_{direction.lower()}.png"
                        )
                    
                    # Save map if available
                    if "map" in new_state["response"]:
                        map_file = self.output_dir / f"map_after_{direction.lower()}.json"
                        with open(map_file, 'w') as f:
                            json.dump(new_state["response"]["map"], f, indent=2)
                    
                    # Use this direction for further testing
                    return direction
                else:
                    print(f"⚠️  Position unchanged after {direction}")
        
        print("⚠️  No direction resulted in movement")
        return None
    
    def generate_report(self):
        """Generate a markdown report of all tests"""
        report_file = self.output_dir.parent / "v3_endpoint_test_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# v3 Server Endpoint Test Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            total = len(self.results)
            successful = sum(1 for r in self.results.values() if r["success"])
            f.write(f"## Summary\n\n")
            f.write(f"- Total endpoints tested: {total}\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- Failed: {total - successful}\n\n")
            
            # Results table
            f.write("## Endpoint Test Results\n\n")
            f.write("| Endpoint | Method | Status | Result |\n")
            f.write("|----------|--------|--------|--------|\n")
            
            for name, result in self.results.items():
                status_icon = "✅" if result["success"] else "❌"
                status_code = result.get("status_code", "N/A")
                f.write(f"| {name} | {result['method']} | {status_code} | {status_icon} |\n")
            
            f.write("\n## Detailed Results\n\n")
            for name, result in self.results.items():
                f.write(f"### {name}\n\n")
                f.write(f"- **Endpoint**: `{result['endpoint']}`\n")
                f.write(f"- **Method**: {result['method']}\n")
                f.write(f"- **Status**: {'✅ Success' if result['success'] else '❌ Failed'}\n")
                if "status_code" in result:
                    f.write(f"- **Status Code**: {result['status_code']}\n")
                if "error" in result:
                    f.write(f"- **Error**: {result['error']}\n")
                if "saved_to" in result:
                    f.write(f"- **Response saved to**: `{result['saved_to']}`\n")
                f.write("\n")
        
        print(f"\n📝 Report saved to {report_file}")
        return str(report_file)


def main():
    tester = EndpointTester()
    
    # Wait for server
    if not tester.wait_for_server():
        print("❌ Server not available, exiting")
        return 1
    
    # Test all endpoints
    tester.test_all_endpoints()
    
    # Test movement
    tester.test_movement()
    
    # Generate report
    tester.generate_report()
    
    print("\n✅ All tests complete!")
    return 0


if __name__ == "__main__":
    exit(main())
