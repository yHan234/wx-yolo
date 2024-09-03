#!/usr/env python3
# -*- coding: UTF-8 -*-

from flask import Flask, request, send_file, make_response
import json
from work import download_and_detect

app = Flask(__name__)


def make_json_response(data, status_code=200):
    response = make_response(json.dumps(data), status_code)
    response.headers["Content-Type"] = "application/json"
    return response


@app.route("/detect", methods=["POST"])
async def detect_api():
    try:
        frames = await download_and_detect(request.get_json()["url"])
    except Exception as e:
        return make_json_response({"err": "处理失败：" + str(e)})

    res = {
        "frames": frames,
        "prompt": "请不要复述目标检测结果，而是根据结果和用户说明推测一个场景，可以适当修饰，不要过度猜想。",
    }

    return make_json_response(res)


@app.route("/ai-plugin.json")
async def plugin_manifest():
    host = request.host_url
    with open("plugin_config/ai-plugin.json", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "application/json"}


@app.route("/openapi.yaml")
async def openapi_spec():
    host = request.host_url
    with open("plugin_config/openapi.yaml", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "text/yaml"}


@app.route("/example.yaml")
async def example_spec():
    host = request.host_url
    with open("plugin_config/example.yaml", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "text/yaml"}


@app.route("/logo.png")
async def plugin_logo():
    return send_file("plugin_config/logo.png", mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
