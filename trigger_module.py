import os
from com_main_module import COMMON_MAIN_MODULE_CLASS
from time import sleep, time
from subprocess import Popen, PIPE
from signal import SIGTERM


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

        web_ui_dir = os.path.join(os.path.dirname(__file__), "web_ui")
        self._print('Initializing trigger module...')
        super().__init__(standalone, config_params, logger_config, web_ui_dir=web_ui_dir)
        self._print('config:\n' + str(self.get_config()) + '\n')

    def main(self):
        config = self.get_config()
        # print('config:\n' + str(config) + '\n')
        # if not os.path.exists(os.path.dirname(config['file_path'])): os.mkdir(os.path.dirname(config['file_path']))
        # self.errors.append('my error')

        p = Popen(['python3', config['trigger_dir'] + '/trigger_main.py'],
                  preexec_fn=os.setsid)
        # p = Popen(['python3', '/var/lib/cloud9/trigger/trigger_main.py'],
        #             stdout=PIPE, shell=True, preexec_fn=os.setsid)

        if self.config.error:
            self.set_config(config)
            self.config.error = None
        while not self.shutdown_event.is_set():
            # set message
            self.message = 'Starting trigger module...'
            sleep(1)
            # read new packets in loop, abort if connection fails or shutdown event is set
            while not self.shutdown_event.is_set():
                # self._print(f'executing module time:{time()} errors:{self.get_errors_list()}')
                # self.errors.append('my error')
                if self.errors:
                    self.message = self.get_errors_list()[-1]
                sleep(1)

        # p.send_signal(SIGTERM)
        os.killpg(os.getpgid(p.pid), SIGTERM)

        self.module_alive = False
        # self.set_config(self.get_config())
        self._print('trigger module thread exited')


