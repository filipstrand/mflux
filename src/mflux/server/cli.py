import argparse
import sys
from . import automatic1111, comfyui, openai_server

def main():
    parser = argparse.ArgumentParser(description="Run mflux server with different API compatibilities")
    parser.add_argument("api", choices=["openai", "automatic1111", "comfyui"], 
                        help="API compatibility mode to run")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    # Add other common arguments here
    
    args = parser.parse_args()
    
    if args.api == "openai":
        openai_server.run_server(host=args.host, port=args.port)
    elif args.api == "automatic1111":
        automatic1111.run_server(host=args.host, port=args.port)
    elif args.api == "comfyui":
        comfyui.run_server(host=args.host, port=args.port)
    else:
        print(f"Unknown API: {args.api}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()