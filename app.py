import os
import subprocess

from flask import Flask, request, jsonify, render_template, send_file
import yaml
import markdown

import utils
from settings import app_config, job_db, tmp_dir, module_file, module_config_type


# load index markdown
with open('templates/index.md', 'r') as f:
    index_content = f.read()

# Flask App
app = Flask(__name__)


# index page
@app.route('/', methods=['GET'])
def home():
    md_template = markdown.markdown(
        index_content, 
        extensions=["fenced_code", "tables"]
        )
    return render_template('index.html', mkd=md_template)

@app.route('/api/run', methods=['POST'])
def run_module():
    # authentication
    auth = request.headers.get('apikey')
    if auth != app_config['apikey']:
        resp = jsonify({'message': 'Unauthorized'})
        resp.status_code = 401
        return resp

    # load data
    data = request.form
    files = request.files

    # check key
    key_list = ['module', 'config']
    error_key = {x: data.get(x) is not None for x in key_list}
    if all(list(error_key.values())) is False:
        error = ', '.join([k for k, v in error_key.items() if v is False])
        resp = jsonify({
            'message': 'Missing keys', 
            'error': error
            })
        resp.status_code = 400
        return resp

    # check if module available
    module = data.get('module')
    if module not in module_file:
        resp = jsonify({'message': 'Module is not available'})
        resp.status_code = 400
        return resp

    # check required file
    file_list = list(files)
    require_file = {k for k, v in module_file[module].items() if v['require'] is True}
    optional_file = {k for k, v in module_file[module].items() if v['require'] is False}
    if require_file.issubset(file_list) is False:
        error = ', '.join([x for x in require_file if x not in file_list])
        resp = jsonify({
            'message': 'Missing files', 
            'error': error
            })
        resp.status_code = 400
        return resp
    # check optional file
    file_list_opt = [x for x in file_list if x not in require_file]
    if len(file_list_opt) > 0 and optional_file.issubset(file_list_opt) is False:
        error = ', '.join([x for x in optional_file if x not in file_list_opt])
        resp = jsonify({
            'message': 'Missing files', 
            'error': error
            })
        resp.status_code = 400
        return resp

    # check file type
    utils.seek_file(files)
    filetype_error = {x: utils.val_filetype(files[x].filename, {module_file[module][x]['filetype']}) for x in file_list}
    if all(list(filetype_error.values())) is False:
        error = ', '.join([k for k, v in filetype_error.items() if v is False])
        resp = jsonify({
            'message': 'File type error', 
            'error': error
            })
        resp.status_code = 400
        return resp

    # check column and data type
    utils.seek_file(files)
    datatype_error = {x: utils.val_datatype(files[x], module_file[module][x]['dtypes']) for x in file_list}
    if all(list(datatype_error.values())) is False:
        error = ', '.join([k for k, v in datatype_error.items() if v is False])
        resp = jsonify({
            'message': 'Data type error/Missing Column', 
            'error': error
            })
        resp.status_code = 400
        return resp

    # check module config format
    config_error = utils.val_config(data.get('config'), module_config_type[module])
    if all(list(config_error.values())) is False:
        error = ', '.join([k for k, v in config_error.items() if v is False])
        resp = jsonify({
            'message': 'Config error', 
            'error': error
            })
        resp.status_code = 400
        return resp

    # create job
    utils.seek_file(files)
    module_config =  utils.format_json(data.get('config'), module_config_type[module])
    job = utils.gen_job_id(app_config['timezone'])
    # check max job and write to database
    print("check maximum job")
    job_del = utils.get_del_job(job_db, app_config['max_job'])
    utils.rm_job(tmp_dir, job_del)
    utils.del_job(job_db, job_del)
    utils.append_job(db=job_db, job_id=job[0], job_dt=job[1], module=module, user=data.get('user'))
    utils.update_status(db=job_db, job_id=job[0], status='start')
    # write input file
    os.mkdir(os.path.join(tmp_dir, job[0]))
    for x in file_list:
        files[x].save(os.path.join(tmp_dir, job[0], module_file[module][x]['filename']))
    with open(os.path.join(tmp_dir, job[0], 'config.yaml'), 'w') as f:
        yaml.dump(module_config, f, sort_keys=False)

    # run module
    proc = subprocess.Popen(["python3", "run.py", module, job[0]])
    utils.update_pid(job_db, job[0], proc.pid)

    # response
    resp = jsonify({
        'message' : 'Module is successfully run in background',
        'job_id': job[0],
        'job_status': '{}/{}'.format(request.url.replace("run", "job"), job[0])
        })
    resp.status_code = 200
    return resp

@app.route('/api/stop', methods=['POST'])
def stop_job():
    # authentication
    auth = request.headers.get('apikey')
    if auth != app_config['apikey']:
        resp = jsonify({'message': 'Unauthorized'})
        resp.status_code = 401
        return resp

    # load data
    data = request.form
 
    # check key
    if data.get('job_id') is None:
        resp = jsonify({
            'message': 'Missing keys', 
            'error': 'job_id'
            })
        resp.status_code = 400
        return resp

    job_id = data.get('job_id')
    # check job id
    if utils.val_jobid(job_id, tmp_dir, job_db) is False:
        resp = jsonify({'message': 'No job id'})
        resp.status_code = 400
        return resp

    # check job status
    if utils.get_job_status(job_db, job_id) != 'running':
        resp = jsonify({'message': 'This job is not currently running'})
        resp.status_code = 400
        return resp

    # stop job
    utils.kill_job(job_db, job_id)
    utils.update_status(db=job_db, job_id=job_id, status='stop')
    # response
    resp = jsonify({
        'message' : 'Stop job',
        'job_id': job_id
        })
    resp.status_code = 200
    return resp

@app.route('/api/jobs', methods=['POST'])
def get_jobs():
    # authentication
    auth = request.headers.get('apikey')
    if auth != app_config['apikey']:
        resp = jsonify({'message': 'Unauthorized'})
        resp.status_code = 401
        return resp

    # response
    jobs_status = utils.get_jobs_status(job_db, request.form.get('user'))
    resp = jsonify(jobs_status)
    resp.status_code = 200
    return resp

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job(job_id):
    # check id
    if utils.val_jobid(job_id, tmp_dir, job_db) is False:
        resp = jsonify({'message': 'No job id'})
        resp.status_code = 400
        return resp

    # find job info
    job_info = utils.get_job_data(job_db, job_id)
    job_info.pop("pid", None)
    job_info['files'] = utils.get_job_files(tmp_dir, job_id)
    job_info['log'] = utils.get_job_log(tmp_dir, job_id, app_config['log_lines'])
    job_info['logs'] = utils.get_job_log(tmp_dir, job_id)
    # response
    resp = jsonify(job_info)
    resp.status_code = 200
    return resp

@app.route('/api/job/<job_id>/<file_name>', methods=['GET'])
def get_file(job_id, file_name):
    # check id
    if utils.val_jobid(job_id, tmp_dir, job_db) is False:
        resp = jsonify({'message': 'No job id'})
        resp.status_code = 400
        return resp

    # file path
    file_path = os.path.join(tmp_dir, job_id, file_name)
    if os.path.isfile(file_path) is False:
        resp = jsonify({'message': 'File is not available'})
        resp.status_code = 400
        return resp
    else:
        return send_file(file_path, attachment_filename=os.path.basename(file_path), as_attachment=True)

@app.route('/api/job/<job_id>/zip', methods=['GET'])
def get_zipfile(job_id):
    # check id
    if utils.val_jobid(job_id, tmp_dir, job_db) is False:
        resp = jsonify({'message': 'No job id'})
        resp.status_code = 400
        return resp

    # file path
    file_path = os.path.join(tmp_dir, job_id, "{}.zip".format(job_id))
    if os.path.isfile(file_path) is False:
        resp = jsonify({'message': 'File is not available'})
        resp.status_code = 400
        return resp
    else:
        return send_file(file_path, attachment_filename=os.path.basename(file_path), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=app_config['web_debug'], host='0.0.0.0', port=5000)
