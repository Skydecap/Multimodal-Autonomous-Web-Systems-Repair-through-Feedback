import argparse

from .dashboard import create_dashboard_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAWSR dashboard server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--target-url", default=None, help="Website URL to test (defaults from .env)")
    parser.add_argument("--source-dir", default=None, help="Source directory for patch preview/apply (defaults from .env)")
    parser.add_argument("--route-prefix", default="", help="Optional route prefix, e.g. /mawsr")
    args = parser.parse_args()

    app = create_dashboard_app(
        target_url=args.target_url,
        source_dir=args.source_dir,
        route_prefix=args.route_prefix,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
