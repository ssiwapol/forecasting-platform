# -*- coding: utf-8 -*-
import os
import subprocess

import yaml
from flask import Flask, request, render_template, jsonify
from flaskext.markdown import Markdown

from mod import FilePath

with open("config.yaml") as f:
    conf = yaml.load(f, Loader=yaml.Loader)
app = Flask(__name__)
Markdown(app, extensions=['fenced_code'])

with open('templates/index.md', 'r') as f:
    index_content = f.read()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', mkd=index_content)
    

@app.route('/api', methods = ['POST'])
def runmodel():
    headers = request.headers
    auth = headers.get("apikey")
    if auth == conf['APIKEY']:
        data = request.get_json()
        fp = FilePath(conf['PLATFORM'], cloud_auth=conf['CLOUD_AUTH'])
        try:
            if data.get('run') != "validate" and data.get('run') != "forecast":
                return jsonify({"message": "ERROR: Run option is not available"}), 400
            elif fp.fileexists(data.get('path')) is False:
                return jsonify({"message": "ERROR: Path file is not avaliable"}), 400
            else:
                if data.get("gbqdest") is None:
                    runsh = './run.sh {} {}'.format(data['run'], data["path"])
                else:
                    runsh = './run.sh {} {} {}'.format(data['run'], data["path"], data["gbqdest"])
                subprocess.call([runsh], shell=True)
                return jsonify({"message": "Successful run in background"}), 200
        except Exception as e:
            return jsonify({"message": "ERROR: {}".format(str(e))}), 400
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


if __name__ == '__main__':
    app.run(debug=conf['DEBUG'], host='0.0.0.0', port=conf['PORT'])
