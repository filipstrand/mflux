import argparse


def main():
    parser = argparse.ArgumentParser(description="Run mflux server with different API compatibilities")
    parser.add_argument("api", choices=["openai", "automatic1111", "comfyui"], 
                        help="API compatibility mode to run")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    
    args = parser.parse_args()
    
    if args.api == "openai":
        from . import openai_server
        openai_server.run_server(host=args.host, port=args.port)
    elif args.api == "automatic1111":
        from . import automatic1111
        automatic1111.run_server(host=args.host, port=args.port)
    elif args.api == "comfyui":
        from . import comfyui
        comfyui.run_server(host=args.host, port=args.port)
    else:
        parser.error(f"Unknown API: {args.api}")

if __name__ == "__main__":
    main()