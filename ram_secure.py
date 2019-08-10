	#!/usr/bin/python3

# Run with nohup in crontab with:
# @reboot python3 $HOME/trafficFlow/prototipo/ram_secure.py > $HOME/logs/output_ram.txt 2>&1

import os
import time
import psutil
import logging
import datetime

logging.basicConfig(filename='ram_secure.txt', filemode='w', format='%(name)s - %(asctime)s : %(message)s',level=logging.INFO)

if __name__ == "__main__":
    PERIOD_SECONDS = 30                  # Report RAM every 10 minutes
    last_time = time.time()
    logging.info("Started RAM Secure at: {}".format(datetime.datetime.now().strftime("%Y-%m-%d")))
    while True:
        porcentajeDeMemoria = psutil.virtual_memory()[2]
        time_now = time.time()
        if (time_now - last_time > PERIOD_SECONDS):
            logging.info("Datetime: {}, memory at {} %".format(datetime.datetime.now().strftime("%H:%M:%S"),str(porcentajeDeMemoria)+'%'))
            last_time = time_now

        if (porcentajeDeMemoria > 92):
            logging.info('Alert memory at {}%'.format(porcentajeDeMemoria))
            os.system('sudo killall -9 python3.5')
        
        if porcentajeDeMemoria > 96:
            logging.info('Urgent Stop on {}% memory overflow'.format(porcentajeDeMemoria))
            os.system('sudo reboot')
