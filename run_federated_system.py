import subprocess
import time
import argparse

NUM_CLIENTS = 5
SERVER_SCRIPT = "server.py"
CLIENT_SCRIPT = "client.py"

def launch_server(model_type):
    print("🚀 Launching Flower Server...")
    subprocess.Popen(
        ["start", "powershell", "-NoExit", "python", SERVER_SCRIPT, "--model_type", model_type],
        shell=True
    )
    print("🧾 Server process launched (PowerShell window should stay open)")

def launch_clients(model_type):
    print(f"\n🚀 Launching {NUM_CLIENTS} Clients using model: {model_type.upper()}...\n")
    for i in range(NUM_CLIENTS):
        print(f"🧠 Launching Client {i} with model: {model_type.upper()}...")
        subprocess.Popen(
            ["start", "powershell", "-NoExit", "python", CLIENT_SCRIPT,
             "--client_id", str(i), "--model_type", model_type],
            shell=True
        )
        time.sleep(2)

    print("\n✅ All clients launched.")
    print("📢 Federated learning system is now running in separate PowerShell windows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["xgb", "catboost", "lr"], default="xgb",
                        help="Model type to use for all clients")
    args = parser.parse_args()

    # 🔥 This line was missing before
    launch_server(args.model_type)
    time.sleep(5)
    launch_clients(args.model_type)