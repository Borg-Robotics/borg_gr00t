import json

import zmq


def main(port=5556):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"Echo server running on port {port}...")

    while True:
        message = socket.recv_json()
        print(f"Received: {json.dumps(message)}")
        socket.send_json({"status": "ok", "received": message})
        print("Sent reply")


if __name__ == "__main__":
    main()
