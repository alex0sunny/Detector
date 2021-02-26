import os, sys
from com_main_module import COMMON_MAIN_MODULE_CLASS
from time import sleep, time
from subprocess import Popen, PIPE
from signal import SIGTERM
import json


# This program is for demonstration purposes only!

class MAIN_MODULE_CLASS(COMMON_MAIN_MODULE_CLASS):
    def __init__(self, trigger_fxn, standalone=False):
        logger_config = {
            'log_file_name': 'trigger_module_log.txt',
            'print_to_stdout': True,
            'stdout_prefix': 'TRIGGER_MODULE'
        }
        config_params = {
            'config_file_name': 'trigger_module_cfg.json',
            'default_config': {'trigger_dir': '/var/lib/cloud9/trigger'}
        }

        web_ui_dir = os.path.join(os.path.dirname(__file__), "backend")
        # self._print('Initializing trigger module...')
        super().__init__(standalone, config_params, logger_config, web_ui_dir=web_ui_dir)
        config = self.get_config()
        # self._print('config:\n' + str(config) + '\n')
        sys.path.append(config['trigger_dir'])

    def custom_web_ui_request(self, in_data):
        path = in_data['path']  # .split('?')
        for ext in ['jpg', 'png', 'ico', 'gif']:
            if path.endswith('.' + ext):
                f = open(self.web_ui_dir + os.sep + path, 'rb')
                data = f.read()
                f.close()
                return {'binary_content': data, 'code': 200,
                        'c_type': 'image/' + ext}
        if in_data['type'] == 'post' and path == 'trigger':
            content = in_data['binary_content']
            dic = json.loads(content.decode())
            # self._print(f'dic:{dic}')
            triggers = dic['triggers']
            triggers = {int(k): 0 if v else 1 for k, v in triggers.items()}
            content = json.dumps({'triggers': triggers}).encode()
            return {'binary_content': content, 'code': 200,
                    'c_type': 'application/json'}

    def main(self):
        workdir = os.path.dirname(__file__)
        config = self.get_config()
        # print('config:\n' + str(config) + '\n')
        # if not os.path.exists(os.path.dirname(config['file_path'])): os.mkdir(os.path.dirname(config['file_path']))
        # self.errors.append('my error')

        # p = Popen(['python3', '/var/lib/cloud9/trigger/trigger_main.py'],
        #             stdout=PIPE, shell=True, preexec_fn=os.setsid)
        if self.config.error:
            self.set_config(config)
            self.config.error = None
        while not self.shutdown_event.is_set():
            # set message
            self.message = 'Starting trigger module...'
            # sleep(1)

            # read new packets in loop, abort if connection fails or shutdown event is set
            while not self.shutdown_event.is_set():
                self.message = 'running'
                if self.errors:
                    self.message = self.get_errors_list()[-1]
                sleep(1)

        self.module_alive = False
        # self.set_config(self.get_config())
        self._print('trigger module thread exited')


