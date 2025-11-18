import modal
import time
import sys

def tail_logs():
    print("Tailing logs for 'poker-bot-training'...")
    cmd = ["modal", "app", "logs", "poker-bot-training"]
    
    import subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("Stopping log tail...")
        process.terminate()

if __name__ == "__main__":
    tail_logs()


