import json
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

def readConfig():
    with open("config.json","r") as file:
        return json.load(file)

class InfluxManager:
    def __init__(self,config):
        self.cfg = config

        self.client = influxdb_client.InfluxDBClient(
            url = self.cfg['url'],
            org = self.cfg['org'],
            token = self.cfg['token']
        )

        self.apiWriter = self.client.write_api(write_options = SYNCHRONOUS)

    def send(self,data):
        task = "monitor_task,host={},location={},room={}".format(self.cfg['host'],self.cfg['location'],self.cfg['room'])
        sequence = [str(task) + "time-detection" + str(data),]
        self.apiWriter.write(bucket = self.cfg['bucket'], org = self.cfg['org'], record = sequence)
        print("Time attention: {}".format(data))