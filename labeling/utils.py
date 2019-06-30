from fastNLP import Callback

import time
import os
import json

start_time = time.time()


def get_schemas(data_path):
    schemas = {}
    with open(os.path.join(data_path, "all_50_schemas"), 'rb') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['subject_type'] + spo['predicate'] + spo['object_type']] = i
    return schemas


class TimingCallback(Callback):
    def on_epoch_end(self):
        print('Sum Time: {:d}s\n\n'.format(round(time.time() - start_time)))

