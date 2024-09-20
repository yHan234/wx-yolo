#!/usr/env python3
# -*- coding: UTF-8 -*-

from flask import Flask, request, make_response
import json
from work import download_and_detect

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
async def detect():
    try:
        frames = await download_and_detect(request.get_json()["url"])
    except Exception as e:
        return {"err": "处理失败：" + str(e)}, 200, {"Content-Type": "application/json"}

    return {"frames": frames, "err": ""}, 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
