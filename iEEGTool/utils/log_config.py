# -*- coding: UTF-8 -*-
'''
@Project ：SEEG_Process 
@File    ：log_config.py
@Author  ：Barry
@Date    ：2021/8/10 20:20 
'''
import logging
import logging.handlers

BASIC_FMT = "%(asctime)s - %(filename)s[line:%(lineno)d] - " \
            "%(levelname)s: %(message)s"
DATA_FMT = '%Y/%m/%d %H:%M:%S'
logging.basicConfig(format=BASIC_FMT,
                    datefmt=DATA_FMT,
                    level=logging.INFO)

def create_logger(logger_name='logger', filename='log.log', save=True) -> object:
    '''
    :param logger_name: str
                        logger name
    :param filename: str
                     filename of log file
    :param save: bool
                 Save to local
    :return:
    '''
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()  # 清除所有Handler，防止logger在控制台重复打印日志

    formatter = logging.Formatter(BASIC_FMT, DATA_FMT)
    if (not logger.handlers) and save:
        # 将日志信息输出到Console(控制台)  好像默认即输出到控制台
        # shandler = logging.StreamHandler()
        # shandler.setFormatter(formatter)
        # shandler.setLevel(logging.INFO)
        # logger.addHandler(shandler)

        # 将日志信息保存到本地
        fhandler = logging.FileHandler(filename, mode='w')
        fhandler.setFormatter(formatter)
        fhandler.setLevel(logging.INFO)
        logger.addHandler(fhandler)

        # 将日志信息保存到本地 SEEG.log SEEG.log1 SEEG.log2 ...
        # rotate_fhandler = logging.handlers.RotatingFileHandler(filename, mode='w',
        #                                         maxBytes=10*1024, backupCount=9)
        # rotate_fhandler.setLevel(logging.INFO)
        # rotate_fhandler.setFormatter(formatter)
        # logger.addHandler(rotate_fhandler)

    return logger
