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


class Logger:
    def __init__(self, logger_id=1):
        self.logger = logging.getLogger(str(logger_id))
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
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

            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


def main():
    logger = Logger().get_logger()
    logger.info(f'Finish Updating Price Data ',
                extra={'module_name': "Stock_price"})


if __name__ == "__main__":
    main()
