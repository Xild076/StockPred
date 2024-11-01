from colorama import Fore
import datetime

class Logging:
    def __init__(self, save_path, active_log=True):
        self.save_path = save_path
        self.active_log = active_log
    
    def _update_log(self, text):
        with open(self.save_path, 'a') as f:
            f.write(text + '\n')
            f.close()

    def success(self, text):
        header = f'[SUCCESS] {datetime.datetime.now().strftime("%m/%d - %H:%M:%S")} === '
        pass_text = Fore.GREEN + header + Fore.RESET + text
        self._update_log(header + text)
        if self.active_log:
            print(pass_text)
    
    def error(self, text):
        header = f'[-ERROR-] {datetime.datetime.now().strftime("%m/%d - %H:%M:%S")} === '
        pass_text = Fore.RED + header + Fore.RESET + text
        self._update_log(header + text)
        if self.active_log:
            print(pass_text)

    def alert(self, text):
        header = f'[-ALERT-] {datetime.datetime.now().strftime("%m/%d - %H:%M:%S")} === '
        pass_text = Fore.YELLOW + header + Fore.RESET + text
        self._update_log(header + text)
        if self.active_log:
            print(pass_text)

    def log(self, text):
        header = f'[LOGGING] {datetime.datetime.now().strftime("%m/%d - %H:%M:%S")} === '
        pass_text = Fore.BLUE + header + Fore.RESET + text
        self._update_log(header + text)
        if self.active_log:
            print(pass_text)
