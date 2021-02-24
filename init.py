import os
import sqlite3

import utils
from settings import tmp_dir, app_config, job_db


def init_db(db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    sql = """
    CREATE TABLE IF NOT EXISTS job (
        id text PRIMARY KEY,
        dt text NOT NULL,
        module text NOT NULL,
        user text,
        pid integer,
        status text
    );"""
    cur.execute(sql)
    conn.commit()
    conn.close()


if __name__=="__main__":
    # create database if not exist
    init_db(job_db)
    # sync job in database and job in directory
    print("sync job")
    job_db_list = list(utils.get_jobs_status(job_db))
    job_dir_list = [os.path.basename(f.path) for f in os.scandir(tmp_dir) if f.is_dir()]
    utils.rm_job(tmp_dir, [x for x in job_dir_list if x not in job_db_list])
    utils.del_job(job_db, [x for x in job_db_list if x not in job_dir_list])
    # check max job
    print("check max job")
    job_del = utils.get_del_job(job_db, app_config['max_job'])
    utils.rm_job(tmp_dir, job_del)
    utils.del_job(job_db, job_del)
