"""
实时监控面板 for YOLO AutoResearch v2

提供实验进度和 GPU 状态的实时监控。
"""
from flask import Flask, jsonify, render_template
import paramiko
import time
import os

app = Flask(__name__)

# H100 服务器配置（需要修改为实际地址）
HOST = os.environ.get("H100_HOST", "YOUR_H100_IP")
USER = os.environ.get("H100_USER", "root")
KEY = os.environ.get("H100_KEY", "/path/to/your/id_rsa")
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/root/yolo-autoresearch-v2")


def ssh_run(cmd: str) -> str:
    """执行 SSH 命令并返回输出"""
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, username=USER, key_filename=KEY, timeout=10)
        _, out, _ = c.exec_command(cmd)
        result = out.read().decode().strip()
        c.close()
        return result
    except Exception as e:
        return f"Error: {e}"


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/status")
def status():
    """获取当前状态"""
    # GPU 状态（每 10 秒轮询）
    try:
        gpu = ssh_run(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,"
            "memory.total,temperature.gpu --format=csv,noheader,nounits"
        )
        if gpu.startswith("Error"):
            gpu_data = {"error": gpu}
        else:
            parts = [x.strip() for x in gpu.split(",")]
            gpu_data = {
                "gpu_util": int(parts[0]) if parts[0].isdigit() else 0,
                "mem_used_mb": int(parts[1]) if parts[1].isdigit() else 0,
                "mem_total_mb": int(parts[2]) if parts[2].isdigit() else 0,
                "temp_c": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
            }
    except:
        gpu_data = {"error": "Unable to fetch GPU status"}

    # 实验结果
    try:
        tsv = ssh_run(f"tail -30 {PROJECT_DIR}/results.tsv")
        rows = [r.split("\t") for r in tsv.strip().split("\n") if r and not r.startswith("exp_id")]
        best = 0.0
        for r in rows:
            if len(r) > 4:
                try:
                    best = max(best, float(r[4]))
                except:
                    pass
        n_experiments = len(rows)
        recent_5 = rows[-5:] if rows else []
    except:
        best = 0.0
        n_experiments = 0
        recent_5 = []

    # 进程状态
    try:
        ps = ssh_run(f"ps aux | grep 'python train.py' | grep -v grep")
        running = bool(ps and "python" in ps)
        if running:
            current_exp = ssh_run(f"ls -td {PROJECT_DIR}/runs/exp_* 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo 'unknown'")
        else:
            current_exp = None
    except:
        running = False
        current_exp = None

    return jsonify({
        "gpu": gpu_data,
        "best_map50": round(best, 4),
        "n_experiments": n_experiments,
        "recent_5": recent_5,
        "training": running,
        "current_exp": current_exp,
        "timestamp": time.time(),
    })


@app.route("/api/results")
def results():
    """获取完整实验结果"""
    try:
        tsv = ssh_run(f"cat {PROJECT_DIR}/results.tsv")
        rows = [r.split("\t") for r in tsv.strip().split("\n") if r]
        return jsonify({"results": rows})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # 本地运行：浏览器打开 http://localhost:8765/api/status
    app.run(port=8765, debug=True)
