# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logger
   Description :
   date：          2024/4/20
-------------------------------------------------
"""
import logging
import colorlog


# 开发一个日志系统， 既要把日志输出到控制台， 还要写入日志文件
class Logger:
    def __init__(self, logger_id=1):
        """
           指定保存日志的文件路径，日志级别，以及调用文件
           将日志存入到指定的文件中
        """
        # 创建一个logger
        self.logger = logging.getLogger(str(logger_id))
        self.logger.setLevel(logging.INFO)

        # 先判断logger是否存在文件句柄，若不存在则进行创建，否则会出现日志重复打印的问题
        if not self.logger.handlers:
            # 创建一个handler，用于写入日志文件
            # fh = logging.FileHandler(logFile)
            # fh.setLevel(logging.DEBUG)

            # 创建彩色日志处理器
            handler = colorlog.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)-8s %(asctime)s | %(module_name)s | %(message_log_color)s%(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'blue',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={
                    'message': {
                        'DEBUG': 'cyan',
                        'INFO': 'white',
                        'WARNING': 'white',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                },
                style='%'
            ))

            # 将处理器添加到日志记录器
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


def main():
    logger = Logger().get_logger()
    # 示例输出
    logger.info(f'Finish Updating Price Data ',
                extra={'module_name': "Stock_price"})


if __name__ == "__main__":
    main()
