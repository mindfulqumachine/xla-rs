import http.server
import socketserver
import json
import subprocess
import os
import tempfile
import sys

PORT = 3001

class PlaygroundHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/evaluate.json':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            code = data.get('code', '')

            print(f"Received code to run:\n{code[:100]}...")

            # Create a temporary directory for the compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                src_file = os.path.join(temp_dir, 'main.rs')
                exe_file = os.path.join(temp_dir, 'main')

                with open(src_file, 'w') as f:
                    f.write(code)

                # Compile and run
                # We assume the server is run from the crate root, so target/debug/deps is available
                compile_cmd = [
                    'rustc', src_file,
                    '-o', exe_file,
                    '-L', 'target/debug/deps',
                    '--edition', '2021'
                ]
                
                # Check for xla_rs library to add --extern if needed, 
                # but usually -L is enough if the code has `extern crate xla_rs;`
                # If the code uses `use xla_rs::...` without extern crate, we might need to force it.
                # But let's rely on the code having `extern crate xla_rs;` which I added.
                
                try:
                    # Compile
                    compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True)
                    if compile_proc.returncode != 0:
                        response = {
                            "success": False,
                            "result": (compile_proc.stderr or "") + (compile_proc.stdout or "")
                        }
                    else:
                        # Run
                        run_proc = subprocess.run([exe_file], capture_output=True, text=True)
                        response = {
                            "success": True,
                            "result": (run_proc.stdout or "") + (run_proc.stderr or "")
                        }
                except Exception as e:
                    response = {
                        "success": False,
                        "result": str(e)
                    }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') # Important for CORS
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

print(f"Starting local playground server on port {PORT}...")
with socketserver.TCPServer(("", PORT), PlaygroundHandler) as httpd:
    httpd.serve_forever()
