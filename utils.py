# -*- coding: utf-8 -*-
import os
import shutil
import signal
import logging
import datetime
import random
import re
import string
import zipfile
import json
import sqlite3

import pandas as pd
from pytz import timezone


class Logging:
    def __init__(self, logpath, tz, stream=False):
        logging.Formatter.converter = self.timetz
        self.tz = tz
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logpath, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        if stream:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

    def timetz(self, *argv):
        return datetime.datetime.now(timezone(self.tz)).timetuple()

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def gen_job_id(tz='Asia/Bangkok'):
    start_time = datetime.datetime.now(timezone(tz))
    job_id = "{}{}".format(start_time.strftime("%Y%m%d%H%M%S"), "".join(random.choice(string.ascii_letters) for i in range(10)))
    job_dt = start_time.strftime("%Y-%m-%d %H:%M:%S")
    return (job_id, job_dt)

# validation utils

def val_datatype(f, dtypes):
    try:
        col = list(dtypes)
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
        dtypes = {k: v for k, v in dtypes.items() if v != 'ds'}
        date_list = [k for k, v in dtypes.items() if v == 'ds']
        pd.read_csv(f, encoding='utf-8', dtype=dtypes, parse_dates=date_list, date_parser=dateparse)[col]
        return True
    except Exception:
        return False

def val_filetype(filename, allow_extension):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allow_extension

def val_jobid(job_id, tmp_dir, job_db):
    conn = sqlite3.connect(job_db)
    cur = conn.cursor()
    sql = """SELECT * FROM job WHERE id = '{}'""".format(job_id)
    row =  cur.execute(sql).fetchone()
    conn.commit()
    conn.close()
    val_db = False if row is None else True
    val_path = os.path.isdir(os.path.join(tmp_dir, job_id))
    return all([val_db, val_path])

def seek_file(files):
    for x in list(files):
        files[x].seek(0)

def val_config(j, conf):
    try:
        j = json.loads(j)
        c = {}
        for i in conf:
            if conf[i] == 'ds':
                try:
                    c[i] = datetime.datetime.strptime(j[i], '%Y-%m-%d')
                    c[i] = True
                except Exception:
                    c[i] = False
            elif type(conf[i]) == list:
                try:
                    c[i] = (j[i] in conf[i])
                except Exception:
                    c[i] = False
            elif type(conf[i]) == dict:
                key_type = list(conf[i])[0]
                val_type = list(conf[i].values())[0]
                try:
                    key_list = [type(key_type(x)) == key_type for x in list(j[i])]
                    val_list = [type(x) == val_type for x in list(j[i].values())]
                    c[i] = all(key_list + val_list)
                except Exception:
                    c[i] = False
            else:
                try:
                    c[i] = (type(j[i]) == conf[i])
                except Exception:
                    c[i] = False
    except Exception:
        c = {'JSON format': False}
    return c


def format_json(j, conf):
    j = json.loads(j)
    c = {}
    for i in conf:
        if conf[i] == 'ds':
            c[i] = datetime.datetime.strptime(j[i], '%Y-%m-%d').date()
        elif type(conf[i]) == dict:
            key_type = list(conf[i])[0]
            c[i] = {key_type(k): v for k,v in j[i].items()}
        else:
            c[i] = j[i]
    return c


# database utils

def append_job(db, job_id, job_dt, module, user=None):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    if user is None:
        sql = """
        INSERT INTO job (id, dt, module)
        VALUES ('{}', '{}', '{}');
        """.format(job_id, job_dt, module)
    else:
        sql = """
        INSERT INTO job (id, dt, module, user)
        VALUES ('{}', '{}', '{}', '{}');
        """.format(job_id, job_dt, module, user)
    cur.execute(sql)
    conn.commit()
    conn.close()

def update_pid(db, job_id, pid):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    sql = """
    UPDATE job
    SET pid = {}
    WHERE id = '{}';
    """.format(pid, job_id)
    cur.execute(sql)
    conn.commit()
    conn.close()

def update_status(db, job_id, status):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    sql = """
    UPDATE job
    SET status = '{}'
    WHERE id = '{}';
    """.format(status, job_id)
    cur.execute(sql)
    conn.commit()
    conn.close()

def get_jobs_status(db, user=None):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    if user is None:
        sql = """SELECT id, status FROM job"""
    else:
        sql = """SELECT id, status FROM job WHERE user = '{}'""".format(user)
    r = {x[0]: x[1] for x in cur.execute(sql).fetchall()}
    conn.commit()
    conn.close()
    return r

def get_job_status(db, job_id):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    sql = """SELECT status FROM job WHERE id = '{}'""".format(job_id)
    r = cur.execute(sql).fetchone()[0]
    conn.commit()
    conn.close()
    return r

def get_job_data(db, job_id):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    sql = """SELECT * FROM job WHERE id = '{}'""".format(job_id)
    row =  cur.execute(sql).fetchone()
    result = dict(zip([x[0] for x in cur.description], row))
    conn.commit()
    conn.close()
    return result

def get_del_job(db, max_job):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    del_row = len(get_jobs_status(db)) - max_job + 1
    sql = """
    SELECT
        id
    FROM
        (SELECT 
            *,
            ROW_NUMBER() OVER(ORDER BY id) row
        FROM job)
    WHERE row <= {}
    """.format(del_row)
    row = [x[0] for x in cur.execute(sql).fetchall()]
    conn.commit()
    conn.close()
    return row

def del_job(db, job_list):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    job_list = ", ".join(["'"+x+"'" for x in job_list])
    sql = """DELETE FROM job WHERE id IN ({})""".format(job_list)
    cur.execute(sql)
    print("delete {} row(s)".format(cur.rowcount))
    conn.commit()
    conn.close()

# job utils

def get_job_files(tmp_dir, job_id):
    job_path = os.path.join(tmp_dir, job_id)
    file_list = [os.path.basename(f.path) for f in os.scandir(job_path) if (f.is_file())]
    return sorted(file_list)

def get_job_log(tmp_dir, job_id, n=None):
    path = os.path.join(tmp_dir, job_id, "logfile.log")
    if os.path.isfile(path):
        with open(path, "r")  as f:
            if n is None:
                r = [row for row in f.readlines()]
            else:
                r = [row for row in f.readlines()][-n:]
        return ''.join(r)
    return None

def rm_job(tmp_dir, job_list):
    for i in job_list:
        del_dir = os.path.join(tmp_dir, i)
        if os.path.isdir(del_dir):
            shutil.rmtree(os.path.join(tmp_dir, i))
    print("remove {} dir(s)".format(len(job_list)))

def zip_output(output_dir, zippath):
    files = [f.path for f in os.scandir(output_dir) if (f.is_file()) and not (bool(re.match(r"[\s\S]+\d+-\d+.csv", f.path)))]
    with zipfile.ZipFile(zippath, 'w') as z:
        for f in files:
            z.write(f, os.path.basename(f))
    return zippath

def kill_job(db, job_id):
    pid = get_job_data(db, job_id)['pid']
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
