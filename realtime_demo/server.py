"""Run the local multi-dataset demo with python -m realtime_demo.server."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from realtime_demo.engine import Conflict
from realtime_demo.workspace import Workspace, user_config


class Handler(SimpleHTTPRequestHandler):
    WEB_ASSETS = {'/', '/index.html', '/style.css', '/app.js', '/roi_visibility.mjs', '/cylinder.mjs'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'realtime_demo/web'), **kwargs)

    def end_headers(self):
        if urlparse(self.path).path in self.WEB_ASSETS:
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):
        if args and not str(args[0]).startswith(('GET /api/state', 'GET /api/frame/')):
            super().log_message(fmt, *args)

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def local_host(self):
        hostname = self.headers.get('Host', '').partition(':')[0].lower()
        return hostname in ('localhost', '127.0.0.1')

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        workspace = self.server.workspace
        try:
            if not self.local_host():
                return self.send_json(dict(error='Use the local demo URL printed by the server.'), 403)
            if path == '/map_visualization/physics.mjs':
                source = ROOT / 'map_visualization/physics.mjs'
                return self.send_bytes(source.read_bytes(), 'text/javascript; charset=utf-8')
            if path == '/api/state':
                return self.send_json(workspace.state())
            if path == '/api/scene':
                return self.send_json(workspace.scene())
            if path.startswith('/api/site-map/'):
                parts = path.split('/')
                if len(parts) != 6:
                    raise ValueError('Invalid site map URL')
                revision = parse_qs(url.query).get('scene', [None])[0]
                engine = workspace.current_engine(int(revision) if revision is not None else None)
                return self.send_bytes(engine.site_map_png(*map(int, parts[3:])), 'image/png')
            if path.startswith('/api/frame/'):
                parts = path.split('/')
                if len(parts) != 6:
                    raise ValueError('Invalid image URL')
                generation, frame, camera = map(int, parts[3:])
                return self.send_bytes(workspace.current_engine().jpeg(generation, frame, camera), 'image/jpeg')
            if path.startswith('/api/'):
                return self.send_json(dict(error='Not found'), 404)
            return super().do_GET()
        except ConnectionError:
            pass
        except (ValueError, OSError) as exc:
            self.send_json(dict(error=str(exc)), 400)

    def do_POST(self):
        try:
            if not self.local_host():
                return self.send_json(dict(error='Use the local demo URL printed by the server.'), 403)
            # Local-only API: reject cross-origin browser mutations.
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                return self.send_json(dict(error='Cross-origin request rejected'), 403)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 8192:
                raise ValueError('Request too large')
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict):
                raise ValueError('Expected JSON object')
            action = urlparse(self.path).path
            workspace = self.server.workspace
            with workspace.mutations:
                return self.apply_action(workspace, action, body)
        except Conflict as exc:
            self.send_json(dict(error=str(exc)), 409)
        except ConnectionError:
            pass
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self.send_json(dict(error=str(exc)), 400)

    def apply_action(self, workspace, action, body):
        engine = workspace.current_engine(body.pop('scene_revision', None))
        if action == '/api/datasets/scan':
            workspace.scan(body.get('directory'))
        elif action == '/api/datasets/refresh':
            workspace.refresh()
        elif action == '/api/datasets/select':
            workspace.select(body.get('dataset'))
        elif action == '/api/cache-directory':
            workspace.cache_directory(body.get('directory'))
        elif action == '/api/cache/generate':
            workspace.generate_cache(body.get('directory'), body.get('fps'))
        elif action == '/api/cache/cancel':
            engine.cancel_cache()
        elif action == '/api/load':
            engine.load(body.get('checkpoint', ''))
        elif action == '/api/benchmark':
            engine.benchmark()
        elif action == '/api/resume':
            engine.resume()
        elif action == '/api/pause':
            engine.pause()
        elif action == '/api/config':
            engine.configure(frame=body.get('frame'), cameras=body.get('cameras'), batch=body.get('batch'), mode=body.get('mode'))
        elif action == '/api/source':
            workspace.select_source(body.get('source'))
        elif action == '/api/mp4-options':
            engine.configure_mp4(body.get('fused_remap'), body.get('prefetch'))
        elif action == '/api/scenarios/save':
            engine.save_scenario(body.get('name'), body.get('start_s'), body.get('end_s'), body.get('id'))
        elif action == '/api/scenarios/delete':
            engine.delete_scenario(body.get('id'))
        elif action == '/api/scenarios/select':
            engine.select_scenario(body['id'])
        else:
            return self.send_json(dict(error='Not found'), 404)
        self.send_json(workspace.state())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8770)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--dataset-root', type=Path, help='scan and save a Dataset folder on startup')
    source.add_argument('--session', type=Path, help='open a legacy WaveTrack recording directly')
    p.add_argument('--config', type=Path, default=user_config(), help='local settings file')
    p.add_argument('--cache-dir', type=Path, help='default cache directory')
    p.add_argument('--seconds', type=float, default=22, help=argparse.SUPPRESS)
    p.add_argument('--input-size', type=int, nargs=2, metavar=('W', 'H'))
    p.add_argument('--prepare', action='store_true', help='load the model and open an existing legacy cache')
    p.add_argument('--checkpoint', help='override the initial BEVNet checkpoint filename')
    p.add_argument('--latency-ms', type=float, default=350)
    args = p.parse_args()
    if args.seconds <= 0 or args.latency_ms <= 0 or (args.input_size and min(args.input_size) < 32):
        p.error('Invalid duration, latency or input size')
    import cv2
    import torch
    # Avoid oversubscription between video decode/OpenCV and tiny CPU tensor operations.
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if args.prepare and not (args.session or args.dataset_root) and (ROOT / 'data/20261107').is_dir():
        args.session = ROOT / 'data/20261107'
    args.config = args.config.expanduser().resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.dataset_root is not None:
        args.dataset_root = args.dataset_root.expanduser().resolve()
    workspace = Workspace(ROOT, args)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.workspace = workspace
    print(f'BEV live demo: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
