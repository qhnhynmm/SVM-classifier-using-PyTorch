import argparse
from utils.get_config import get_config
from task.classify_task import Classify_Task
from task.clustering_task import Clustering_Task
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required = True)

args = parser.parse_args()

config = get_config(args.config_file)

if config.task == 'classify':
    task = Classify_Task(config)
if config.task == 'clustering':
    task = Clustering_Task(config)
    
task.training() #traning, khi nào muốn predict thì cmt lại
task.evaluate() #đánh giá trên test data