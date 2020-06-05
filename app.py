# -*- coding: utf-8 -*-
import os
import subprocess

import yaml
from flask import Flask, request, render_template, jsonify
import markdown

from utils import FilePath


tmp_dir = "tmp"
config_path = os.path.join(tmp_dir, "config.yaml")
with open(config_path) as f:
    conf = yaml.load(f, Loader=yaml.Loader)
cloudauth_path = os.path.join(tmp_dir, conf['CLOUD_AUTH']) if conf['CLOUD_AUTH'] else None
with open('templates/index.md', 'r') as f:
    index_content = f.read()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    md_template = markdown.markdown(
        index_content, 
        extensions=["fenced_code", "tables"]
        )
    return render_template('index.html', mkd=md_template)
    

@app.route('/api', methods = ['POST'])
def runmodel():
    headers = request.headers
    auth = headers.get("apikey")
    if auth == conf['APIKEY']:
        data = request.get_json()
        fp = FilePath(conf['PLATFORM'], cloud_auth=cloudauth_path)
        try:
            if data.get('run') != "validate" and data.get('run') != "forecast":
                return jsonify({"message": "ERROR: Run option is not available"}), 400
            elif fp.fileexists(os.path.join(tmp_dir, data.get('path'))) is False:
                return jsonify({"message": "ERROR: Path file is not avaliable"}), 400
            else:
                runsh = './run.sh {} {}'.format(data['run'], data["path"])
                subprocess.call([runsh], shell=True)
                return jsonify({"message": "Successful run in background"}), 200
        except Exception as e:
            return jsonify({"message": "ERROR: {}".format(str(e))}), 400
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


if __name__ == '__main__':
    app.run(debug=conf['DEBUG'], host='0.0.0.0', port=5000)
