#! /bin/bash
# Run prototipo in startup 

# to configure crontab execute:
# @reboot sh $HOME/trafficFlow/prototipo/secure_run.sh > $HOME/logs/output_full.txt 2>&1
echo "Killing processes"
sudo killall -9 python3.5
cd $HOME/trafficFlow/object-detection-server/
echo "Launching program"
nohup python3.5 monitorPlates.py -s True &
nohup python3.5 -u ram_secure.py &
echo "Running succesfully"

cd
